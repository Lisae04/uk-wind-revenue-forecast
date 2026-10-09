"""Day-ahead wind generation: physical power-curve model + ML residual correction.

Level from GWA, dynamics from metered data
-----------------------------------------
* The physical model's long-term level is anchored to the capstone's Global Wind Atlas
  means (k_c = GWA_c / mean ERA5 hub wind), so annual energy stays consistent with GA.
* The ML model learns what the power curve gets wrong hour by hour (ramp timing,
  high-wind cut-out, curtailment, icing, direction effects) from a metered reference:
      r_t = CF_obs,t - CF_phys,t        (both on the reference farm)
  and adds r_hat to the portfolio:  CF_corr = clip(CF_phys + r_hat, 0, 1 - losses).
* The reference farm's own physical model is level-matched on training data only, so
  r_t has mean ~0 there and the correction moves shape, not the GWA-anchored level.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from .timeutils import utc_ts
from .power import air_density, cell_energy, choose_iec_class, density_equivalent, \
    power_norm, shear_alpha, to_hub

KINDS = ("act", "fc")


# ------------------------------------------------------------------ physical model
def hub_wind(w: pd.DataFrame, kind: str, tcfg) -> np.ndarray:
    return to_hub(w[f"ws100_{kind}"], w[f"ws10_{kind}"],
                  float(tcfg["hub_height_m"]), float(tcfg["gwa_height_m"]))


def downscale_factors(cells: pd.DataFrame, weather: dict, tcfg) -> pd.Series:
    """k_c = GWA long-term mean / ERA5 long-term mean at the farm's NWP point."""
    k = {}
    for c in cells.itertuples():
        u = hub_wind(weather[c.farm_id], "act", tcfg)
        m = np.nanmean(u)
        if not np.isfinite(m) or m <= 0:
            raise ValueError(f"ERA5 hub wind for farm {c.farm_id} is empty")
        k[c.cell_id] = c.gwa_ms / m
    return pd.Series(k, name="k")


def portfolio_physical(cells, farms, weather: dict, tcfg) -> dict[str, pd.DataFrame]:
    """Hourly MWh per farm for actual (ERA5) and day-ahead forecast weather."""
    k = downscale_factors(cells, weather, tcfg)
    out = {}
    for kind in KINDS:
        cols = {}
        for f in farms.itertuples():
            w = weather[f.farm_id]
            u = hub_wind(w, kind, tcfg)
            rho = air_density(w[f"t2m_{kind}"], w[f"sp_{kind}"])
            e = np.zeros(len(w))
            for c in cells[cells.farm_id == f.farm_id].itertuples():
                e += cell_energy(k[c.cell_id] * u, rho, c.iec_class, tcfg)
            cols[f.name] = e
        out[kind] = pd.DataFrame(cols, index=next(iter(weather.values())).index)
    out["k"] = k
    return out


def long_term_check(phys_act: pd.DataFrame, farms: pd.DataFrame) -> pd.DataFrame:
    """Compare ERA5-driven capacity factor with the GA farm CF (same turbine, same losses)."""
    rows = []
    for f in farms.itertuples():
        cf = phys_act[f.name].mean() / f.mw
        rows.append({"farm": f.name, "mw": f.mw, "ga_cf": f.ga_cf, "model_cf": cf,
                     "ratio": cf / f.ga_cf,
                     "model_aep_gwh": cf * f.mw * 8760 / 1000, "ga_aep_gwh": f.ga_aep_gwh})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ reference site
def reference_cf(w: pd.DataFrame, kind: str, k: float, iec_class: str, tcfg) -> np.ndarray:
    u = k * hub_wind(w, kind, tcfg)
    rho = air_density(w[f"t2m_{kind}"], w[f"sp_{kind}"])
    if tcfg.get("air_density_correction", True):
        u = density_equivalent(u, rho)
    return power_norm(u, tcfg["power_curve_dir"], iec_class) * (1 - float(tcfg["losses"]))


def level_match(w, cf_obs: pd.Series, train_mask, iec_class, tcfg) -> float:
    """Find k so that mean physical CF (actual weather) equals mean observed CF on train."""
    m = train_mask & cf_obs.notna().to_numpy()
    if m.sum() < 24 * 30:
        raise ValueError("Need at least 30 days of observed reference output before test_start")
    target = cf_obs[m].mean()
    lo, hi = 0.3, 3.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if reference_cf(w[m], "act", mid, iec_class, tcfg).mean() < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ------------------------------------------------------------------ ML correction
def correction_features(w: pd.DataFrame, cf_phys_fc, tcfg, tz="Europe/London") -> pd.DataFrame:
    """Only day-ahead information: forecast weather + calendar."""
    u = hub_wind(w, "fc", tcfg)
    wd = np.deg2rad(w["wd100_fc"].to_numpy(dtype=float))
    local = w.index.tz_convert(tz)
    return pd.DataFrame({
        "cf_phys": np.asarray(cf_phys_fc, float),
        "ws_hub": u,
        "alpha": shear_alpha(w["ws100_fc"], w["ws10_fc"]),
        "wd_sin": np.sin(wd), "wd_cos": np.cos(wd),
        "t2m": w["t2m_fc"].to_numpy(dtype=float),
        "rho": air_density(w["t2m_fc"], w["sp_fc"]),
        "hour": local.hour, "doy_sin": np.sin(2 * np.pi * local.dayofyear / 365.25),
        "doy_cos": np.cos(2 * np.pi * local.dayofyear / 365.25),
        "dow": local.dayofweek,
    }, index=w.index)


def walk_forward_correction(X: pd.DataFrame, r: pd.Series, test_start, retrain_days: int,
                            params: dict, X_apply: dict[str, pd.DataFrame] | None = None):
    """Expanding-window LightGBM on the residual r; refit every `retrain_days`.

    Returns out-of-sample residual predictions on the reference (from test_start) and,
    for each frame in X_apply, predictions made with the model available at that time.
    """
    test_start = utc_ts(test_start)
    end = X.index.max()
    pred_ref = pd.Series(np.nan, index=X.index)
    pred_apply = {k: pd.Series(np.nan, index=v.index) for k, v in (X_apply or {}).items()}
    model = None
    t = test_start
    while t <= end:
        t_next = t + pd.Timedelta(days=retrain_days)
        train = (X.index < t) & r.notna().to_numpy() & X.notna().all(axis=1).to_numpy()
        model = LGBMRegressor(**params).fit(X[train], r[train])
        win = (X.index >= t) & (X.index < t_next)
        ok = win & X.notna().all(axis=1).to_numpy()
        pred_ref[ok] = model.predict(X[ok])
        for k, Xa in (X_apply or {}).items():
            wa = (Xa.index >= t) & (Xa.index < t_next) & Xa.notna().all(axis=1).to_numpy()
            pred_apply[k][wa] = model.predict(Xa[wa])
        t = t_next
    return pred_ref, pred_apply, model


def climatology_cf(cf_obs: pd.Series, train_mask, tz="Europe/London") -> pd.Series:
    """Baseline: mean observed CF by (month, local hour) on the training period."""
    local = cf_obs.index.tz_convert(tz)
    key = pd.MultiIndex.from_arrays([local.month, local.hour])
    tab = cf_obs[train_mask].groupby([local.month[train_mask], local.hour[train_mask]]).mean()
    return pd.Series(tab.reindex(key).to_numpy(), index=cf_obs.index)


def reference_class(mean_ws: float, tcfg) -> str:
    return choose_iec_class(mean_ws, tcfg["iec_class_vave_ms"])
