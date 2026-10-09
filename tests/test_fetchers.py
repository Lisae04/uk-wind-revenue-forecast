"""Offline tests of the fetchers with payloads shaped like the real API responses."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ukwind import fetch_elexon as fe  # noqa: E402
from ukwind import fetch_weather as fw  # noqa: E402


def test_mid_prices_volume_weighted(monkeypatch, tmp_path):
    rows = [  # same fields as /datasets/MID/stream
        {"startTime": "2025-01-01T01:00:00Z", "dataProvider": "APXMIDP", "settlementDate": "2025-01-01",
         "settlementPeriod": 3, "price": 10.0, "volume": 100.0},
        {"startTime": "2025-01-01T01:30:00Z", "dataProvider": "APXMIDP", "settlementDate": "2025-01-01",
         "settlementPeriod": 4, "price": 20.0, "volume": 300.0},
        {"startTime": "2025-01-01T01:00:00Z", "dataProvider": "N2EXMIDP", "settlementDate": "2025-01-01",
         "settlementPeriod": 3, "price": 0.0, "volume": 0.0},
        {"startTime": "2025-01-01T02:00:00Z", "dataProvider": "APXMIDP", "settlementDate": "2025-01-01",
         "settlementPeriod": 5, "price": 99.0, "volume": 0.0},
    ]
    monkeypatch.setattr(fe, "get_json", lambda url, params, cache: rows)
    p = fe.fetch_prices("2025-01-01", "2025-01-01", tmp_path)
    assert p.loc["2025-01-01 01:00+00:00", "price"] == 17.5      # (10*100 + 20*300) / 400
    assert np.isnan(p.loc["2025-01-01 02:00+00:00", "price"])     # zero volume -> missing
    assert len(p) == 24


def test_windfor_vintage_and_hourly(monkeypatch, tmp_path):
    rows = [  # /datasets/WINDFOR/stream fields
        {"dataset": "WINDFOR", "publishTime": "2025-01-14T08:00:00Z", "startTime": "2025-01-15T10:00:00Z", "generation": 5000},
        {"dataset": "WINDFOR", "publishTime": "2025-01-14T08:00:00Z", "startTime": "2025-01-15T10:30:00Z", "generation": 7000},
        {"dataset": "WINDFOR", "publishTime": "2025-01-14T11:00:00Z", "startTime": "2025-01-15T10:00:00Z", "generation": 1},
    ]
    monkeypatch.setattr(fe, "get_json", lambda url, params, cache: rows)
    s = fe.fetch_wind_forecast("2025-01-15", "2025-01-15", tmp_path, "Europe/London", "09:00")
    assert s.loc["2025-01-15 10:00+00:00"] == 6000                # after-gate vintage ignored


def test_b1610_hourly_sum(monkeypatch, tmp_path):
    rows = [{"dataset": "B1610", "psrType": "Wind Onshore", "bmUnit": "T_X-1", "nationalGridBmUnitId": "X-1",
             "settlementDate": "2025-06-01", "settlementPeriod": 1,
             "halfHourEndTime": "2025-06-01T00:30:00Z", "quantity": 10.0},
            {"dataset": "B1610", "psrType": "Wind Onshore", "bmUnit": "T_X-1", "nationalGridBmUnitId": "X-1",
             "settlementDate": "2025-06-01", "settlementPeriod": 2,
             "halfHourEndTime": "2025-06-01T01:00:00Z", "quantity": 12.0}]
    monkeypatch.setattr(fe, "get_json", lambda url, params, cache: rows)
    s = fe.fetch_bmu_output("T_X-1", "2025-05-31", "2025-06-01", tmp_path)
    assert s.loc["2025-06-01 00:00+00:00"] == 22.0                # periods starting 00:00 and 00:30
    assert np.isnan(s.loc["2025-06-01 01:00+00:00"])               # an hour needs both half-hours


def test_previous_runs_day1_day2_switch(monkeypatch, tmp_path):
    t = pd.date_range("2025-01-15", periods=24, freq="h").strftime("%Y-%m-%dT%H:%M").tolist()
    hourly = {"time": t}
    for k in fw.VARS:
        hourly[f"{k}_previous_day1"] = [1.0] * 24
        hourly[f"{k}_previous_day2"] = [2.0] * 24
    monkeypatch.setattr(fw, "get_json", lambda url, params, cache: {"hourly": hourly})
    df = fw.fetch_forecasts(55.0, -2.0, "2025-01-15", "2025-01-15", tmp_path)
    # GMT: day1 run issued at valid-24h is before D-1 09:00 only for valid hours <= 09:00
    assert (df.loc[:"2025-01-15 09:00+00:00", "ws100_fc"] == 1.0).all()
    assert (df.loc["2025-01-15 10:00+00:00":, "ws100_fc"] == 2.0).all()
