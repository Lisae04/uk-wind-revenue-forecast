"""Day-ahead GB price forecasting: naive baselines, LEAR and LightGBM + SHAP.

Information set for delivery day D (gate closure D-1 09:00 London):
    * Elexon day-ahead demand forecast (NDF) and wind forecast (WINDFOR), latest vintage
      published before the gate;
    * prices known before the gate: everything up to D-1 08:59 local. A within-day index
      such as MID is NOT known for D-1 afternoon/evening at 09:00, so the classic
      "same hour yesterday" lag would leak. We use 48 h and 168 h lags and the D-1
      morning mean instead;
    * calendar (local hour, weekday, month, English bank holidays).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import LassoLarsIC
from sklearn.preprocessing import StandardScaler

from .timeutils import cutoff_for_times, utc_ts

try:
    import holidays
    _HOL = holidays.country_holidays("GB", subdiv="ENG")
except Exception:  # pragma: no cover - optional dependency
    _HOL = None


# ------------------------------------------------------------------ dataset
def build_dataset(system: pd.DataFrame, tz="Europe/London", gate="09:00") -> pd.DataFrame:
    need = {"price", "demand_fc_mw", "wind_fc_mw"}
    if not need <= set(system.columns):
        raise KeyError(f"system table lacks {need - set(system.columns)}")
    df = pd.DataFrame(index=system.index)
    df["price"] = system["price"]
    df["demand_fc"] = system["demand_fc_mw"]
    df["wind_fc"] = system["wind_fc_mw"]
    df["resid_fc"] = df["demand_fc"] - df["wind_fc"]
    df["wind_share_fc"] = df["wind_fc"] / df["demand_fc"].where(df["demand_fc"] > 0)

    local = df.index.tz_convert(tz)
    df["hour"] = local.hour
    df["dow"] = local.dayofweek
    df["month"] = local.month
    df["weekend"] = (local.dayofweek >= 5).astype(int)
    dates = local.date
    df["holiday"] = [int(d in _HOL) if _HOL is not None else 0 for d in dates]

    # price lags that are known before the gate closure
    df["p_lag48"] = df["price"].shift(48)
    df["p_lag168"] = df["price"].shift(168)
    day = pd.Series(local.normalize().tz_localize(None), index=df.index)
    # hours ending at least one hour before the gate (MID is published after each period)
    morning = df["price"].where(local.hour < int(gate.split(":")[0]) - 1)
    m_mean = morning.groupby(day.values).mean()
    df["p_d1_morning"] = (day - pd.Timedelta(days=1)).map(m_mean).to_numpy()
    df["p_d2_mean"] = (day - pd.Timedelta(days=2)).map(df["price"].groupby(day.values).mean()).to_numpy()
    df["resid_lag168"] = df["resid_fc"].shift(168)
    df.attrs["cutoff"] = cutoff_for_times(df.index, tz, gate)
    return df


FEATURES = ["demand_fc", "wind_fc", "resid_fc", "wind_share_fc", "hour", "dow", "month",
            "weekend", "holiday", "p_lag48", "p_lag168", "p_d1_morning", "p_d2_mean",
            "resid_lag168"]


def assert_no_lookahead(df: pd.DataFrame, tz="Europe/London", gate="09:00") -> None:
    """Every price-derived feature must come from a timestamp before the gate closure."""
    cutoff = cutoff_for_times(df.index, tz, gate)
    src48 = df.index - pd.Timedelta(hours=48)
    src168 = df.index - pd.Timedelta(hours=168)
    if not ((src48 < cutoff).all() and (src168 < cutoff).all()):
        raise AssertionError("price lag uses information after gate closure")


# ------------------------------------------------------------------ baselines
def naive_forecasts(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {"naive_D-2": df["p_lag48"], "naive_D-7": df["p_lag168"]}


# ------------------------------------------------------------------ LEAR
class _Asinh:
    """Variance-stabilising transform (Uniejewski et al., 2018): asinh((p - med) / MAD)."""

    def fit(self, p):
        p = np.asarray(p, float)
        self.med = np.nanmedian(p)
        mad = np.nanmedian(np.abs(p - self.med)) / 0.6745
        self.mad = mad if mad > 0 else 1.0
        return self

    def fwd(self, p):
        return np.arcsinh((np.asarray(p, float) - self.med) / self.mad)

    def inv(self, z):
        return np.sinh(z) * self.mad + self.med


LEAR_FEATS = ["demand_fc", "wind_fc", "resid_fc", "wind_share_fc", "weekend", "holiday",
              "p_lag48", "p_lag168", "p_d1_morning", "p_d2_mean", "resid_lag168"]


def lear_forecast(df: pd.DataFrame, test_start, window_days=365, every_days=7) -> pd.Series:
    """One LASSO per local hour, AIC-selected penalty, rolling window, periodic recalibration."""
    test_start = utc_ts(test_start)
    pred = pd.Series(np.nan, index=df.index)
    price_cols = ["p_lag48", "p_lag168", "p_d1_morning", "p_d2_mean"]
    dow = pd.get_dummies(df["dow"], prefix="dow", dtype=float)
    X_all = pd.concat([df[LEAR_FEATS], dow], axis=1)
    t = test_start
    while t <= df.index.max():
        t_next = t + pd.Timedelta(days=every_days)
        tr_mask = (df.index < t) & (df.index >= t - pd.Timedelta(days=window_days))
        for h in range(24):
            hm = (df["hour"] == h).to_numpy()
            tr = tr_mask & hm & X_all.notna().all(axis=1).to_numpy() & df["price"].notna().to_numpy()
            te = (df.index >= t) & (df.index < t_next) & hm & X_all.notna().all(axis=1).to_numpy()
            if tr.sum() < 60 or te.sum() == 0:
                continue
            tf = _Asinh().fit(df.loc[tr, "price"])
            X = X_all.copy()
            for c in price_cols:                       # transform price features the same way
                X[c] = tf.fwd(X[c])
            sc = StandardScaler().fit(X[tr])
            model = LassoLarsIC(criterion="aic").fit(sc.transform(X[tr]), tf.fwd(df.loc[tr, "price"]))
            pred[te] = tf.inv(model.predict(sc.transform(X[te])))
        t = t_next
    return pred


# ------------------------------------------------------------------ LightGBM
def lgbm_forecast(df: pd.DataFrame, test_start, every_days=30, params=None):
    test_start = utc_ts(test_start)
    params = {"objective": "l1", **(params or {})}
    pred = pd.Series(np.nan, index=df.index)
    ok_x = df[FEATURES].notna().all(axis=1).to_numpy()
    model = None
    t = test_start
    while t <= df.index.max():
        t_next = t + pd.Timedelta(days=every_days)
        tr = (df.index < t) & ok_x & df["price"].notna().to_numpy()
        te = (df.index >= t) & (df.index < t_next) & ok_x
        model = LGBMRegressor(**params).fit(df.loc[tr, FEATURES], df.loc[tr, "price"])
        if te.any():
            pred[te] = model.predict(df.loc[te, FEATURES])
        t = t_next
    return pred, model


def shap_importance(model, X: pd.DataFrame, n=3000, seed=0):
    import shap
    Xs = X.dropna().sample(min(n, len(X.dropna())), random_state=seed)
    sv = shap.TreeExplainer(model).shap_values(Xs)
    imp = pd.Series(np.abs(sv).mean(axis=0), index=Xs.columns).sort_values(ascending=False)
    return imp.rename("mean_abs_shap_gbp_mwh"), sv, Xs
