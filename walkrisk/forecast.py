"""Step 3. Predict next month's elderly pedestrian crashes per dong and pick the top 10.

Inputs for target month t (all from months before t):
    x_log_expo        log(1 + elderly walking exposure) in t-1
    x_prev_crash      crashes in t-1
    x_prev_obs        observation rate in t-1
    x_cum_mean_crash  mean monthly crashes from the first month up to t-1

Raw exposure left a spatial pattern in the residuals (Moran's I 0.361 -> 0.065 after
the log). A month dummy was dropped: it extrapolated to unseen months and inflated the
total. Poisson, negative binomial and LightGBM scored the same top-10 capture (19.0%),
so the final model is the plain Poisson GLM.

Evaluation: for each target month, retrain on every earlier target month, then compare
the model's top 10 against a baseline that ranks dongs by past crashes. Ties are broken
by previous-month crashes (descending), then dong code (ascending).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

X = ["x_log_expo", "x_prev_crash", "x_prev_obs", "x_cum_mean_crash"]
Y, T = "y_next_crash", "target_month"


def model_frame(panel: pd.DataFrame, *, dong_col="EMD_CD", month_col="month") -> pd.DataFrame:
    """Turn the dong x month panel into one row per (dong, target month) with lagged features."""
    p = panel.sort_values([dong_col, month_col]).copy()
    g = p.groupby(dong_col)
    p["x_log_expo"] = np.log1p(g["exposure"].shift(1))
    p["x_prev_crash"] = g["crash"].shift(1)
    p["x_prev_obs"] = g["obs_rate"].shift(1)
    p["x_cum_mean_crash"] = g["crash"].transform(lambda s: s.shift(1).expanding().mean())
    p = p.rename(columns={"crash": Y, month_col: T})
    return p.dropna(subset=X)[[dong_col, T, *X, Y]].reset_index(drop=True)


def fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Standardise with the training mean/std, fit a Poisson GLM, predict the test rows."""
    mean = train[X].mean()
    std = train[X].std(ddof=0)
    assert (std > 0).all()
    x_train = sm.add_constant(((train[X] - mean) / std).to_numpy(), has_constant="add")
    x_test = sm.add_constant(((test[X] - mean) / std).to_numpy(), has_constant="add")
    model = sm.GLM(train[Y].to_numpy(float), x_train, family=sm.families.Poisson()).fit(maxiter=200)
    assert model.converged
    return model.predict(x_test)


def monthly_retrain(frame: pd.DataFrame, target_months, *, train_from=None) -> pd.DataFrame:
    """Predict each target month with a model trained only on earlier target months."""
    if train_from is not None:
        frame = frame[frame[T] >= train_from]
    results = []
    for month in target_months:
        train = frame[frame[T] < month]
        test = frame[frame[T] == month].copy()
        test["prediction"] = fit_predict(train, test)
        test["baseline"] = test["x_cum_mean_crash"]
        results.append(test)
    return pd.concat(results, ignore_index=True)


def capture_at_k(evaluation: pd.DataFrame, score: str, *, k: int = 10, dong_col: str = "EMD_CD") -> float:
    """Share of all evaluated crashes that fall in each month's top-k dongs by `score`."""
    captured = 0
    for _, group in evaluation.groupby(T):
        ranked = group.sort_values([score, "x_prev_crash", dong_col], ascending=[False, False, True])
        captured += ranked.head(k)[Y].sum()
    return captured / evaluation[Y].sum()


def scores(evaluation: pd.DataFrame, *, k: int = 10, dong_col: str = "EMD_CD") -> dict:
    y, yhat = evaluation[Y].to_numpy(float), evaluation["prediction"].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(y > 0, y * np.log(y / yhat), 0.0) - (y - yhat)
    return {
        "mae": float(np.mean(np.abs(y - yhat))),
        "poisson_deviance": float(2 * np.mean(term)),
        "total_ratio": float(yhat.sum() / y.sum()),
        f"capture@{k}_model": capture_at_k(evaluation, "prediction", k=k, dong_col=dong_col),
        f"capture@{k}_baseline": capture_at_k(evaluation, "baseline", k=k, dong_col=dong_col),
    }
