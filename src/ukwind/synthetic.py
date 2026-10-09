"""Synthetic stand-in data with the same schema as the real fetchers.

Used for tests, CI and a first offline run. Numbers produced from it are NOT findings:
the structure is plausible (Weibull winds, correlated sites, forecast errors that grow
with wind, merit-order prices that fall with wind share), but every parameter is invented.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from .timeutils import hourly_index


def _ar1(n, phi, sigma, rng):
    e = rng.normal(0, sigma, n)
    x = np.empty(n)
    x[0] = e[0] / np.sqrt(1 - phi ** 2)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def _weibull_from_normal(z, mean_ws, k=2.0):
    lam = mean_ws / math.gamma(1 + 1 / k)        # Weibull scale giving the target mean
    u = np.clip(stats.norm.cdf(z), 1e-6, 1 - 1e-6)
    return lam * (-np.log(1 - u)) ** (1 / k)


def make_weather(idx, mean_ws, common, rng, fc_err=1.1):
    n = len(idx)
    own = _ar1(n, 0.97, np.sqrt(1 - 0.97 ** 2), rng)
    z = 0.8 * common + 0.6 * own
    z = z / z.std()
    doy = idx.dayofyear.to_numpy()
    season = 1 + 0.15 * np.cos(2 * np.pi * (doy - 15) / 365.25)
    ws100 = _weibull_from_normal(z, mean_ws) * season
    hour = idx.hour.to_numpy()
    alpha = np.clip(0.16 + 0.06 * np.cos(2 * np.pi * (hour - 3) / 24) + rng.normal(0, 0.03, n), 0.02, 0.45)
    ws10 = ws100 * (10 / 100) ** alpha
    wd = (240 + np.cumsum(rng.normal(0, 6, n))) % 360
    t2m = 10 - 6 * np.cos(2 * np.pi * (doy - 20) / 365.25) + 3 * np.sin(2 * np.pi * (hour - 9) / 24) \
        + _ar1(n, 0.95, 0.6, rng)
    sp = 1010 + _ar1(n, 0.995, 0.8, rng)

    err = _ar1(n, 0.9, 0.45, rng) * (0.6 + 0.08 * ws100) * fc_err
    ws100_fc = np.clip(ws100 + err, 0, None)
    return pd.DataFrame({
        "ws100_act": ws100, "ws10_act": ws10, "wd100_act": wd, "t2m_act": t2m, "sp_act": sp,
        "ws100_fc": ws100_fc, "ws10_fc": ws100_fc * (10 / 100) ** alpha,
        "wd100_fc": (wd + rng.normal(0, 15, n)) % 360,
        "t2m_fc": t2m + rng.normal(0, 1.0, n), "sp_fc": sp + rng.normal(0, 1.5, n),
    }, index=idx)


def make_all(cfg, cells: pd.DataFrame, farms: pd.DataFrame, seed: int = 7):
    """Return weather per farm, system table, reference sites and synthetic GWA means."""
    from .power import power_norm

    rng = np.random.default_rng(seed)
    p = cfg["period"]
    idx = hourly_index(p["start"], p["end"])
    n = len(idx)
    common = _ar1(n, 0.985, np.sqrt(1 - 0.985 ** 2), rng)

    weather = {}
    for f in farms.itertuples():
        weather[f.farm_id] = make_weather(idx, 0.92 * f.gwa_ms, common, rng)

    # ---- system: national wind, demand, price
    u_nat = _weibull_from_normal(common / common.std(), 8.6)
    cap = np.linspace(14_500, 16_500, n)
    cf_nat = power_norm(u_nat, cfg["turbine"]["power_curve_dir"], "II") * 0.9
    wind_out = cap * cf_nat
    wind_fc = np.clip(wind_out + _ar1(n, 0.9, 0.04, rng) * cap * 0.6, 0, None)

    local = idx.tz_convert("Europe/London")
    h, doy, dow = local.hour.to_numpy(), local.dayofyear.to_numpy(), local.dayofweek.to_numpy()
    daily = np.interp(h, [0, 4, 7, 12, 17, 19, 23], [0.78, 0.70, 0.88, 0.95, 1.05, 1.0, 0.82])
    demand = (27_000 + 5_000 * np.cos(2 * np.pi * (doy - 15) / 365.25)) * daily
    demand *= np.where(dow >= 5, 0.9, 1.0)
    demand_fc = demand + rng.normal(0, 500, n)

    gas = 85 + np.cumsum(rng.normal(0, 0.15, n))              # slow fuel-cost drift
    resid = demand - wind_out
    price = 0.55 * gas * np.clip(resid / 22_000, 0.05, None) ** 1.3 + _ar1(n, 0.8, 6, rng)
    price = np.where(resid < 7_000, price - 35, price)          # negative-price episodes
    price += np.where(rng.random(n) < 0.004, rng.uniform(80, 250, n), 0)   # scarcity spikes
    vol = np.where(rng.random(n) < 0.002, 0, rng.uniform(800, 2500, n))
    system = pd.DataFrame({"price": np.where(vol > 0, price, np.nan), "volume": vol,
                           "demand_fc_mw": demand_fc, "wind_fc_mw": wind_fc,
                           "wind_outturn_mw": wind_out}, index=idx)

    # ---- one metered reference farm (behaves like a real BMU: curtailment, icing, outages)
    ref_w = make_weather(idx, 8.8, common, rng)
    u = 1.03 * ref_w["ws100_act"].to_numpy()
    cf = power_norm(u, cfg["turbine"]["power_curve_dir"], "I") * (1 - 0.12)
    curtail = (h <= 5) & (u > 11)
    cf = np.where(curtail, np.minimum(cf, 0.55), cf)
    icing = (ref_w["t2m_act"].to_numpy() < 0.5) & (u > 5)
    cf = np.where(icing, cf * 0.7, cf)
    day_id = pd.factorize(idx.floor("D"))[0]
    out_days = set(rng.choice(np.unique(day_id), size=25, replace=False))
    cf = np.where(np.isin(day_id, list(out_days)), cf * 0.5, cf)
    cf = np.clip(cf + rng.normal(0, 0.02, n), 0, 1)
    refs = {"SYN-REF-1": {"weather": ref_w, "cf_obs": pd.Series(cf, index=idx),
                          "capacity_mw": 60.0, "lat": 55.2, "lon": -2.9}}
    return weather, system, refs


def synthetic_gwa(cells: pd.DataFrame, seed: int = 3) -> pd.Series:
    """Plausible 100 m means by region, only for running without england_wind.csv."""
    rng = np.random.default_rng(seed)
    base = {"TF": 8.4, "SK": 8.1, "NY": 9.4, "NZ": 9.0}
    return pd.Series({c: base.get(c[:2], 8.5) + rng.normal(0, 0.15) for c in cells}, name="wind_speed_ms")
