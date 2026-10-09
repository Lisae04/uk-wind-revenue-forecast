"""Forecast metrics and the Diebold-Mariano test."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def metrics(y, yhat) -> dict:
    y, yhat = pd.Series(y, dtype=float), pd.Series(yhat, dtype=float)
    m = y.notna() & yhat.notna()
    y, yhat = y[m].to_numpy(), yhat[m].to_numpy()
    if len(y) == 0:
        return {"n": 0, "MAE": np.nan, "RMSE": np.nan, "sMAPE_%": np.nan, "bias": np.nan}
    e = y - yhat
    den = np.abs(y) + np.abs(yhat)
    with np.errstate(divide="ignore", invalid="ignore"):
        smape = np.mean(np.where(den > 0, 2 * np.abs(e) / den, 0)) * 100
    return {"n": int(len(y)), "MAE": float(np.mean(np.abs(e))),
            "RMSE": float(np.sqrt(np.mean(e ** 2))), "sMAPE_%": float(smape),
            "bias": float(np.mean(yhat - y))}


def metrics_table(y: pd.Series, preds: dict, scale_ref: str | None = None) -> pd.DataFrame:
    """Rows = models. rMAE = MAE / MAE(scale_ref), the relative MAE of Lago et al. (2021)."""
    common = y.notna()
    for p in preds.values():
        common &= p.reindex(y.index).notna()
    rows = {k: metrics(y[common], p.reindex(y.index)[common]) for k, p in preds.items()}
    df = pd.DataFrame(rows).T
    if scale_ref in df.index:
        df["rMAE"] = df["MAE"] / df.loc[scale_ref, "MAE"]
    return df


def dm_test(y: pd.Series, p1: pd.Series, p2: pd.Series, tz="Europe/London") -> tuple[float, float]:
    """Multivariate DM test on daily loss differentials (Lago et al., 2021, eq. 8).

    d_D = ||e1_D||_1 / 24 - ||e2_D||_1 / 24.  H1: model 2 is more accurate than model 1.
    Returns (DM statistic, one-sided p-value). p < 0.05 -> model 2 significantly better.
    """
    df = pd.DataFrame({"y": y, "a": p1, "b": p2}).dropna()
    day = df.index.tz_convert(tz).normalize()
    d = (df["y"] - df["a"]).abs().groupby(day).mean() - (df["y"] - df["b"]).abs().groupby(day).mean()
    d = d.dropna()
    if len(d) < 10 or d.std(ddof=1) == 0:
        return np.nan, np.nan
    dm = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    return float(dm), float(1 - stats.norm.cdf(dm))


def dm_matrix(y, preds: dict) -> pd.DataFrame:
    names = list(preds)
    out = pd.DataFrame(np.nan, index=names, columns=names)
    for a in names:
        for b in names:
            if a != b:
                out.loc[a, b] = dm_test(y, preds[a], preds[b])[1]
    out.index.name = "p-value: column better than row"
    return out
