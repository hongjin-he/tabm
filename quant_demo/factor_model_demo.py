"""
TabM applied to a synthetic cross-sectional factor model (quant finance demo).

Why synthetic data: this repo ships no market data, and pulling real data would
add an external dependency / API key requirement. The synthetic generator below
mimics a standard cross-sectional factor setup (N assets x T dates x K factors,
noisy nonlinear target = forward return) so the pipeline is representative of a
real factor-to-return regression task. Swap `make_synthetic_factor_data` for a
loader over your own factor panel (e.g. Barra-style exposures, alpha factors)
to use this for real.

Usage:
    python quant_demo/factor_model_demo.py
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, NamedTuple, Optional

import numpy as np
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import torch
import torch.nn as nn

import tabm


def make_synthetic_factor_data(
    n_assets: int = 500,
    n_dates: int = 400,
    n_factors: int = 12,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a panel of (asset, date) factor exposures and forward returns.

    The target is a nonlinear, noisy combination of the factors, which is a
    reasonable stand-in for "true" alpha in a factor model: linear models and
    GBDTs both leave some signal on the table, which is where TabM-style
    tabular deep learning can help.
    """
    rng = np.random.default_rng(seed)
    n = n_assets * n_dates
    x = rng.normal(size=(n, n_factors)).astype(np.float32)

    w = rng.normal(size=n_factors) * np.array(
        [1.0 / (i + 1) for i in range(n_factors)]
    )
    linear_term = x @ w
    nonlinear_term = 0.5 * np.tanh(x[:, 0] * x[:, 1]) + 0.3 * np.sin(x[:, 2])
    noise = rng.normal(scale=1.5, size=n)

    y = (linear_term + nonlinear_term + noise).astype(np.float32)
    return x, y


class RegressionLabelStats(NamedTuple):
    mean: float
    std: float


def main() -> None:
    seed = 0
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    X_num, Y = make_synthetic_factor_data()
    n_num_features = X_num.shape[1]

    all_idx = np.arange(len(Y))
    trainval_idx, test_idx = sklearn.model_selection.train_test_split(
        all_idx, train_size=0.8, random_state=seed
    )
    train_idx, val_idx = sklearn.model_selection.train_test_split(
        trainval_idx, train_size=0.8, random_state=seed
    )

    data_numpy = {
        "train": {"x_num": X_num[train_idx], "y": Y[train_idx]},
        "val": {"x_num": X_num[val_idx], "y": Y[val_idx]},
        "test": {"x_num": X_num[test_idx], "y": Y[test_idx]},
    }

    # Quantile transform, as recommended for TabM's numerical features.
    noise = np.random.default_rng(0).normal(
        0.0, 1e-5, data_numpy["train"]["x_num"].shape
    ).astype(np.float32)
    preprocessing = sklearn.preprocessing.QuantileTransformer(
        n_quantiles=max(min(len(train_idx) // 30, 1000), 10),
        output_distribution="normal",
        subsample=10**9,
    ).fit(data_numpy["train"]["x_num"] + noise)
    for part in data_numpy:
        data_numpy[part]["x_num"] = preprocessing.transform(
            data_numpy[part]["x_num"]
        ).astype(np.float32)

    Y_train = data_numpy["train"]["y"].copy()
    regression_label_stats = RegressionLabelStats(
        Y_train.mean().item(), Y_train.std().item()
    )
    Y_train = (Y_train - regression_label_stats.mean) / regression_label_stats.std

    data = {
        part: {k: torch.as_tensor(v, device=device) for k, v in part_data.items()}
        for part, part_data in data_numpy.items()
    }
    for part in data:
        data[part]["y"] = data[part]["y"].float()
    Y_train_t = torch.as_tensor(Y_train, device=device).float()

    model = tabm.TabM.make(
        n_num_features=n_num_features,
        cat_cardinalities=[],
        d_out=1,
        num_embeddings=None,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=3e-4)
    gradient_clipping_norm: Optional[float] = 1.0

    def apply_model(part: str, idx: torch.Tensor) -> torch.Tensor:
        return model(data[part]["x_num"][idx], None).squeeze(-1).float()

    def loss_fn(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_pred = y_pred.flatten(0, 1)
        y_true = y_true.repeat_interleave(model.backbone.k)
        return nn.functional.mse_loss(y_pred, y_true)

    @torch.inference_mode()
    def evaluate(part: str) -> tuple[float, float]:
        """Returns (RMSE, cross-sectional rank-IC) on the original label scale."""
        model.eval()
        eval_batch_size = 8096
        y_pred = (
            torch.cat(
                [
                    apply_model(part, idx)
                    for idx in torch.arange(
                        len(data[part]["y"]), device=device
                    ).split(eval_batch_size)
                ]
            )
            .cpu()
            .numpy()
        )
        y_pred = y_pred * regression_label_stats.std + regression_label_stats.mean
        y_pred = y_pred.mean(axis=1)  # average TabM's k sub-models
        y_true = data[part]["y"].cpu().numpy()

        rmse = float(sklearn.metrics.mean_squared_error(y_true, y_pred) ** 0.5)
        # Rank-IC: Spearman correlation between predicted and realized forward returns,
        # the standard quality metric for a cross-sectional alpha factor.
        from scipy.stats import spearmanr

        rank_ic = float(spearmanr(y_pred, y_true).correlation)
        return rmse, rank_ic

    rmse0, ic0 = evaluate("test")
    print(f"Before training  | test RMSE: {rmse0:.4f} | test Rank-IC: {ic0:.4f}")

    train_size = len(train_idx)
    batch_size = 256
    patience = 16
    remaining_patience = patience
    best_val = -math.inf
    best_state: dict[str, Any] = deepcopy(model.state_dict())

    for epoch in range(200):
        model.train()
        for batch_idx in torch.randperm(train_size, device=device).split(batch_size):
            optimizer.zero_grad()
            loss = loss_fn(apply_model("train", batch_idx), Y_train_t[batch_idx])
            loss.backward()
            torch.nn.utils.clip_grad.clip_grad_norm_(
                model.parameters(), gradient_clipping_norm
            )
            optimizer.step()

        val_rmse, val_ic = evaluate("val")
        val_score = -val_rmse
        improved = val_score > best_val
        print(
            f'{"*" if improved else " "} epoch {epoch:<3} '
            f"val RMSE {val_rmse:.4f}  val Rank-IC {val_ic:.4f}"
        )
        if improved:
            best_val = val_score
            best_state = deepcopy(model.state_dict())
            remaining_patience = patience
        else:
            remaining_patience -= 1
        if remaining_patience < 0:
            break

    model.load_state_dict(best_state)
    test_rmse, test_ic = evaluate("test")
    print("\n[Summary]")
    print(f"test RMSE:    {test_rmse:.4f}")
    print(f"test Rank-IC: {test_ic:.4f}  (higher is better; this is a synthetic sanity check, not a real backtest)")


if __name__ == "__main__":
    main()
