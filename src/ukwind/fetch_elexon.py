"""GB market and system data from the Elexon Insights API (no key needed).

Datasets
    MID      Market Index price per settlement period (APXMIDP)      -> price target
    NDF      National Demand forecast, day-ahead vintages             -> price feature
    WINDFOR  Transmission-metered wind forecast, many vintages        -> price feature
    FUELHH   Generation by fuel, outturn (WIND)                       -> national reference CF
    B1610    Metered output per BM unit                               -> reference farm CF
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .http import get_json
from .timeutils import hourly_index, select_vintage

BASE = "https://data.elexon.co.uk/bmrs/api/v1"


def _rows(js) -> list:
    if isinstance(js, list):
        return js
    if isinstance(js, dict) and isinstance(js.get("data"), list):
        return js["data"]
    raise RuntimeError(f"Unexpected Elexon payload: {str(js)[:200]}")


def _windows(start, end, days):
    s = pd.Timestamp(start, tz="UTC")
    end = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    while s < end:
        e = min(s + pd.Timedelta(days=days), end)
        yield s, e
        s = e


def _iso(t):
    return t.strftime("%Y-%m-%dT%H:%MZ")


def _to_hourly_mean(s: pd.Series) -> pd.Series:
    """Half-hourly average MW (or price) -> hourly mean (MW == MWh for one hour)."""
    return s.sort_index().resample("h").mean()


# ---------------------------------------------------------------- prices
def fetch_prices(start, end, cache: Path, provider="APXMIDP") -> pd.DataFrame:
    rows = []
    for s, e in _windows(start, end, 7):
        rows += _rows(get_json(f"{BASE}/datasets/MID/stream",
                               {"from": _iso(s), "to": _iso(e), "dataProviders": provider},
                               cache / "elexon_mid"))
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No MID data returned")
    df = df[df["dataProvider"] == provider].copy()
    df["startTime"] = pd.to_datetime(df["startTime"], utc=True)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    df = df.drop_duplicates("startTime").set_index("startTime").sort_index()
    # zero-volume periods carry no traded price; treat them as missing
    df.loc[df["volume"] <= 0, "price"] = np.nan
    df["pv"] = df["price"] * df["volume"]
    h = df[["pv", "volume"]].resample("h").sum(min_count=1)
    out = pd.DataFrame({"price": h["pv"] / h["volume"].replace(0, np.nan),
                        "volume": h["volume"]})
    return out.reindex(hourly_index(start, end))


# ---------------------------------------------------------------- forecasts (vintaged)
def _fetch_vintaged(dataset, value, start, end, cache, tz, gate, chunk_days=2):
    rows = []
    s0 = pd.Timestamp(start) - pd.Timedelta(days=3)       # D-1 vintages for the first days
    for s, e in _windows(s0, end, chunk_days):
        rows += _rows(get_json(f"{BASE}/datasets/{dataset}/stream",
                               {"publishDateTimeFrom": _iso(s), "publishDateTimeTo": _iso(e)},
                               cache / f"elexon_{dataset.lower()}"))
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No {dataset} data returned")
    df[value] = pd.to_numeric(df[value], errors="coerce")
    v = select_vintage(df, [value], tz, gate)
    return _to_hourly_mean(v[value]).reindex(hourly_index(start, end))


def fetch_demand_forecast(start, end, cache, tz, gate):
    return _fetch_vintaged("NDF", "demand", start, end, cache, tz, gate)


def fetch_wind_forecast(start, end, cache, tz, gate):
    return _fetch_vintaged("WINDFOR", "generation", start, end, cache, tz, gate)


# ---------------------------------------------------------------- outturn
def fetch_wind_outturn(start, end, cache: Path) -> pd.Series:
    rows = []
    for s, e in _windows(start, end, 14):
        rows += _rows(get_json(f"{BASE}/datasets/FUELHH/stream", {
            "settlementDateFrom": s.date().isoformat(),
            "settlementDateTo": (e - pd.Timedelta(days=1)).date().isoformat(),
            "fuelType": "WIND"}, cache / "elexon_fuelhh"))
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No FUELHH data returned")
    df = df[df["fuelType"] == "WIND"].copy()
    df["startTime"] = pd.to_datetime(df["startTime"], utc=True)
    df["generation"] = pd.to_numeric(df["generation"], errors="coerce")
    s = df.drop_duplicates("startTime").set_index("startTime")["generation"]
    return _to_hourly_mean(s).reindex(hourly_index(start, end)).rename("wind_outturn_mw")


def fetch_bmu_output(bmu: str, start, end, cache: Path) -> pd.Series:
    """Metered MWh per hour for one BM unit (B1610 quantity is MWh per half-hour)."""
    rows = []
    for s, e in _windows(start, end, 7):
        rows += _rows(get_json(f"{BASE}/datasets/B1610/stream",
                               {"from": _iso(s), "to": _iso(e), "bmUnit": bmu},
                               cache / "elexon_b1610"))
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No B1610 data for {bmu}")
    df["end"] = pd.to_datetime(df["halfHourEndTime"], utc=True)
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    s = df.drop_duplicates("end").set_index("end")["quantity"]
    s.index = s.index - pd.Timedelta(minutes=30)               # label by period start
    h = s.sort_index().resample("h").sum(min_count=2)           # need both half-hours
    return h.reindex(hourly_index(start, end)).rename(bmu)


def fetch_bmu_reference(cache: Path) -> pd.DataFrame:
    return pd.DataFrame(_rows(get_json(f"{BASE}/reference/bmunits/all", {}, cache / "elexon_ref")))


def fetch_system(cfg) -> pd.DataFrame:
    p, m = cfg["period"], cfg["market"]
    cache = cfg["paths"]["cache"]
    df = fetch_prices(p["start"], p["end"], cache, m["price_provider"])
    df["demand_fc_mw"] = fetch_demand_forecast(p["start"], p["end"], cache, m["tz"], m["gate_closure_local"])
    df["wind_fc_mw"] = fetch_wind_forecast(p["start"], p["end"], cache, m["tz"], m["gate_closure_local"])
    df["wind_outturn_mw"] = fetch_wind_outturn(p["start"], p["end"], cache)
    return df
