"""Hourly weather per site from Open-Meteo (no API key).

* Actuals  : ERA5 reanalysis, archive API.
* Forecast : archived NWP runs, Previous Runs API (archive starts Jan 2024).
             `*_previous_day1` = value predicted 24 h before valid time,
             `*_previous_day2` = 48 h before. For each delivery hour we use day1 only
             if that run was issued before the D-1 gate closure, otherwise day2.

Output columns (UTC hourly index):
    ws100_act, ws10_act, wd100_act, t2m_act, sp_act,
    ws100_fc,  ws10_fc,  wd100_fc,  t2m_fc,  sp_fc
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .http import get_json
from .timeutils import cutoff_for_times, hourly_index

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
PREVIOUS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
VARS = {"wind_speed_100m": "ws100", "wind_speed_10m": "ws10",
        "wind_direction_100m": "wd100", "temperature_2m": "t2m", "surface_pressure": "sp"}


def _chunks(start, end, months=3):
    s = pd.Timestamp(start)
    end = pd.Timestamp(end)
    while s <= end:
        e = min(s + pd.DateOffset(months=months) - pd.Timedelta(days=1), end)
        yield s.date().isoformat(), e.date().isoformat()
        s = e + pd.Timedelta(days=1)


def _frame(js) -> pd.DataFrame:
    h = js.get("hourly") or {}
    if "time" not in h:
        raise RuntimeError(f"Open-Meteo response without hourly data: {str(js)[:200]}")
    df = pd.DataFrame(h)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.set_index("time")


def fetch_actuals(lat, lon, start, end, cache: Path) -> pd.DataFrame:
    parts = []
    for s, e in _chunks(start, end):
        js = get_json(ARCHIVE_URL, {
            "latitude": round(lat, 4), "longitude": round(lon, 4),
            "start_date": s, "end_date": e, "hourly": ",".join(VARS),
            "wind_speed_unit": "ms", "timezone": "UTC", "models": "era5",
        }, cache / "openmeteo_archive")
        parts.append(_frame(js))
    df = pd.concat(parts).sort_index()
    df = df[~df.index.duplicated()].rename(columns={k: f"{v}_act" for k, v in VARS.items()})
    return df


def fetch_forecasts(lat, lon, start, end, cache: Path, tz="Europe/London", gate="09:00",
                    model: str = "best_match") -> pd.DataFrame:
    names = [f"{k}_previous_day{d}" for k in VARS for d in (1, 2)]
    parts = []
    for s, e in _chunks(start, end):
        params = {"latitude": round(lat, 4), "longitude": round(lon, 4),
                  "start_date": s, "end_date": e, "hourly": ",".join(names),
                  "wind_speed_unit": "ms", "timezone": "UTC"}
        if model != "best_match":
            params["models"] = model
        parts.append(_frame(get_json(PREVIOUS_URL, params, cache / "openmeteo_previous")))
    raw = pd.concat(parts).sort_index()
    raw = raw[~raw.index.duplicated()]

    issued_day1 = raw.index - pd.Timedelta(hours=24)
    use_day1 = np.asarray(issued_day1 <= cutoff_for_times(raw.index, tz, gate))
    out = pd.DataFrame(index=raw.index)
    for k, v in VARS.items():
        d1 = pd.to_numeric(raw.get(f"{k}_previous_day1"), errors="coerce")
        d2 = pd.to_numeric(raw.get(f"{k}_previous_day2"), errors="coerce")
        out[f"{v}_fc"] = np.where(use_day1 & d1.notna(), d1, d2)
    return out


def fetch_site_weather(lat, lon, cfg, cache: Path) -> pd.DataFrame:
    p, m = cfg["period"], cfg["market"]
    act = fetch_actuals(lat, lon, p["start"], p["end"], cache)
    fc = fetch_forecasts(lat, lon, p["start"], p["end"], cache, m["tz"], m["gate_closure_local"])
    df = act.join(fc, how="outer").reindex(hourly_index(p["start"], p["end"]))
    return clean_weather(df)


def clean_weather(df: pd.DataFrame, max_gap_h: int = 3) -> pd.DataFrame:
    """Physical bounds, short-gap interpolation, and a report of what was filled."""
    df = df.copy()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if c.startswith(("ws100", "ws10")):
            df.loc[(df[c] < 0) | (df[c] > 60), c] = np.nan
        elif c.startswith("wd100"):
            df.loc[(df[c] < 0) | (df[c] > 360), c] = np.nan
        elif c.startswith("t2m"):
            df.loc[(df[c] < -40) | (df[c] > 45), c] = np.nan
        elif c.startswith("sp"):
            df.loc[(df[c] < 850) | (df[c] > 1090), c] = np.nan
    n_before = int(df.isna().sum().sum())
    lin = [c for c in df.columns if not c.startswith("wd100")]
    df[lin] = df[lin].interpolate(limit=max_gap_h, limit_area="inside")
    wd = [c for c in df.columns if c.startswith("wd100")]   # angles: no linear interpolation
    df[wd] = df[wd].ffill(limit=max_gap_h)
    df.attrs["filled"] = n_before - int(df.isna().sum().sum())
    df.attrs["still_missing"] = int(df.isna().sum().sum())
    return df
