"""Market-day and information-set helpers (GB, gate closure on D-1)."""
from __future__ import annotations

import pandas as pd


def to_utc_index(idx) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(idx)
    return idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")


def delivery_day(ts_utc, tz: str = "Europe/London") -> pd.Series:
    """Local calendar date of each delivery timestamp (as tz-naive midnight)."""
    ts = pd.DatetimeIndex(to_utc_index(ts_utc)).tz_convert(tz)
    return pd.Series(ts.normalize().tz_localize(None), index=to_utc_index(ts_utc))


def cutoff_utc(days, tz: str = "Europe/London", gate: str = "09:00") -> pd.DatetimeIndex:
    """Gate closure for delivery day D = D-1 at `gate` local time, returned in UTC."""
    days = pd.DatetimeIndex(days).normalize()
    if days.tz is not None:
        days = days.tz_localize(None)
    hh, mm = (int(x) for x in gate.split(":"))
    local = days - pd.Timedelta(days=1) + pd.Timedelta(hours=hh, minutes=mm)
    return local.tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward").tz_convert("UTC")


def cutoff_for_times(ts_utc, tz: str = "Europe/London", gate: str = "09:00") -> pd.DatetimeIndex:
    """Information cut-off applicable to each delivery timestamp."""
    return cutoff_utc(delivery_day(ts_utc, tz).values, tz, gate)


def select_vintage(df: pd.DataFrame, value_cols, tz="Europe/London", gate="09:00",
                   time_col="startTime", pub_col="publishTime") -> pd.DataFrame:
    """Keep, for every delivery time, the latest forecast published before its gate closure.

    This is what makes the forecast features day-ahead honest: a forecast issued after
    D-1 09:00 can never be used for a delivery hour on day D.
    """
    if df.empty:
        return pd.DataFrame(columns=list(value_cols))
    d = df[[time_col, pub_col, *value_cols]].copy()
    d[time_col] = to_utc_index(pd.to_datetime(d[time_col], utc=True))
    d[pub_col] = to_utc_index(pd.to_datetime(d[pub_col], utc=True))
    d["cutoff"] = cutoff_for_times(d[time_col], tz, gate)
    d = d[d[pub_col] <= d["cutoff"]]
    d = d.sort_values([time_col, pub_col]).drop_duplicates(time_col, keep="last")
    return d.set_index(time_col)[list(value_cols)].sort_index()


def hourly_index(start, end) -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(start, tz="UTC"),
                         pd.Timestamp(end, tz="UTC") + pd.Timedelta(hours=23),
                         freq="h", name="time")


def utc_ts(x) -> pd.Timestamp:
    """Timestamp in UTC whether or not the input already carries a timezone."""
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")
