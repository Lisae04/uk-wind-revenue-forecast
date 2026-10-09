"""Merchant revenue at the GB reference price.

    Revenue_t        = E_t [MWh] * P_t [GBP/MWh]
    Capture price    = sum(E_t P_t) / sum(E_t)
    Capture rate     = capture price / baseload price (simple mean of P_t)
    Forecast value   = how far day-ahead revenue expectations are from realised revenue
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def capture_stats(gen: pd.Series, price: pd.Series) -> dict:
    d = pd.DataFrame({"e": gen, "p": price}).dropna()
    if d.empty or d["e"].sum() <= 0:
        return {"energy_mwh": 0.0, "revenue_gbp": 0.0, "capture_price": np.nan,
                "baseload_price": np.nan, "capture_rate": np.nan}
    rev = float((d["e"] * d["p"]).sum())
    cap = rev / d["e"].sum()
    base = float(d["p"].mean())
    return {"energy_mwh": float(d["e"].sum()), "revenue_gbp": rev, "capture_price": cap,
            "baseload_price": base,
            # the ratio is meaningless when baseload is near zero
            "capture_rate": cap / base if abs(base) >= 10 else np.nan,
            "capture_discount": base - cap,
            "energy_at_negative_price_mwh": float(d.loc[d["p"] < 0, "e"].sum()),
            "negative_price_hours": int((d["p"] < 0).sum())}


def by_period(gen: pd.Series, price: pd.Series, freq="MS", tz="Europe/London") -> pd.DataFrame:
    d = pd.DataFrame({"e": gen, "p": price}).dropna()
    key = d.index.tz_convert(tz).tz_localize(None).to_period(freq[0]).to_timestamp()
    return pd.DataFrame({k: capture_stats(g["e"], g["p"]) for k, g in d.groupby(key)}).T


def daily_revenue_error(gen_fc, price_fc, gen_act, price_act, tz="Europe/London") -> pd.DataFrame:
    d = pd.DataFrame({"exp": gen_fc * price_fc, "real": gen_act * price_act}).dropna()
    day = d.index.tz_convert(tz).normalize()
    daily = d.groupby(day).sum()
    daily["error"] = daily["exp"] - daily["real"]
    return daily


def cannibalisation(price: pd.Series, wind_share: pd.Series) -> dict:
    """OLS slope of price on forecast wind share of demand (GBP/MWh per +10 pp)."""
    d = pd.DataFrame({"p": price, "w": wind_share}).dropna()
    if len(d) < 100:
        return {"slope_gbp_per_10pp": np.nan, "r2": np.nan}
    b, a = np.polyfit(d["w"], d["p"], 1)
    r2 = np.corrcoef(d["w"], d["p"])[0, 1] ** 2
    return {"slope_gbp_per_10pp": float(b * 0.10), "intercept": float(a), "r2": float(r2)}
