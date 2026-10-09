import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ukwind import power, price, timeutils  # noqa: E402
from ukwind.config import load_config  # noqa: E402
from ukwind.grid import bng_to_latlon, gridref_to_bng  # noqa: E402


# ------------------------------------------------------------------ grid
def test_gridref_centre():
    assert gridref_to_bng("TF8630") == (586500.0, 330500.0)
    assert gridref_to_bng("NZ0605", centre=False) == (406000.0, 505000.0)


def test_farm_centres_match_ga_export():
    farms = pd.read_csv(ROOT / "data/reference/ga_portfolio_farms.csv")
    for f in farms.itertuples():
        en = np.array([gridref_to_bng(c) for c in f.cells.split(";")])
        assert abs(en[:, 0].mean() - f.centre_e) < 1.0
        assert abs(en[:, 1].mean() - f.centre_n) < 1.0


def test_latlon_norfolk():
    lat, lon = bng_to_latlon(586500, 330500)
    assert 52.8 < lat < 52.9 and 0.7 < lon < 0.85


def test_bad_gridref():
    with pytest.raises(ValueError):
        gridref_to_bng("XX12")
    with pytest.raises(ValueError):
        gridref_to_bng("TF863")


# ------------------------------------------------------------------ power
CURVES = ROOT / "data/reference/power_curves"


def test_power_curve_bounds():
    ws = np.array([0, 2.9, 3, 12, 25, 25.1, np.nan])
    p = power.power_norm(ws, CURVES, "II")
    assert p[0] == 0 and p[1] == 0 and p[-1] == 0 and p[5] == 0
    assert 0 < p[2] < 0.05 and p[3] == pytest.approx(1.0, abs=0.05)
    assert np.all((p >= 0) & (p <= 1))


def test_iec_class_choice():
    vave = {"III": 7.5, "II": 8.5, "I": 10.0}
    assert power.choose_iec_class(7.2, vave) == "III"
    assert power.choose_iec_class(8.4, vave) == "II"
    assert power.choose_iec_class(9.6, vave) == "I"


def test_shear_and_density():
    u = np.array([8.0, 10.0])
    assert np.allclose(power.to_hub(u, u * 0.7, 100, 100), u)
    up = power.to_hub(u, u * 0.7, 120, 100)
    assert np.all(up > u)
    rho = power.air_density(15.0, 1013.25)
    assert rho == pytest.approx(1.225, abs=0.005)


# ------------------------------------------------------------------ gate closure
def test_cutoff_bst_and_gmt():
    summer = timeutils.cutoff_utc(["2025-06-15"])[0]
    winter = timeutils.cutoff_utc(["2025-01-15"])[0]
    assert summer == pd.Timestamp("2025-06-14 08:00", tz="UTC")
    assert winter == pd.Timestamp("2025-01-14 09:00", tz="UTC")


def test_select_vintage_drops_late_publications():
    df = pd.DataFrame({
        "startTime": ["2025-01-15T12:00Z"] * 3,
        "publishTime": ["2025-01-13T12:00Z", "2025-01-14T08:30Z", "2025-01-14T10:00Z"],
        "demand": [1.0, 2.0, 3.0]})
    v = timeutils.select_vintage(df, ["demand"])
    assert v["demand"].iloc[0] == 2.0     # 10:00 publication is after the 09:00 gate


# ------------------------------------------------------------------ no look-ahead
def _system(n_days=40, seed=0):
    rng = np.random.default_rng(seed)
    idx = timeutils.hourly_index("2025-03-01", pd.Timestamp("2025-03-01") + pd.Timedelta(days=n_days))
    return pd.DataFrame({"price": rng.normal(50, 20, len(idx)),
                         "demand_fc_mw": rng.normal(28000, 3000, len(idx)),
                         "wind_fc_mw": rng.normal(8000, 3000, len(idx))}, index=idx)


@pytest.mark.parametrize("day", ["2025-03-20", "2025-03-30", "2025-04-02"])   # incl. DST change
def test_price_features_ignore_post_gate_prices(day):
    sys_df = _system()
    base = price.build_dataset(sys_df)
    D = pd.Timestamp(day)
    rows = timeutils.delivery_day(base.index) == D
    cut = timeutils.cutoff_utc([D])[0]
    shocked = sys_df.copy()
    shocked.loc[shocked.index >= cut, "price"] = 1e6        # rewrite everything after gate
    feats = price.build_dataset(shocked)
    cols = [c for c in price.FEATURES if c.startswith("p_")]
    pd.testing.assert_frame_equal(base.loc[rows.values, cols], feats.loc[rows.values, cols])


# ------------------------------------------------------------------ end to end (synthetic)
def test_pipeline_synthetic(tmp_path):
    cfg_dict = yaml.safe_load(open(ROOT / "config.yaml"))
    cfg_dict["period"] = {"start": "2024-01-01", "end": "2024-06-30", "test_start": "2024-05-01"}
    for k in ("cache", "processed", "results"):
        cfg_dict["paths"][k] = str(tmp_path / k)
    cfg_dict["generation_model"]["lgbm"]["n_estimators"] = 50
    cfg_dict["price_model"]["lgbm"]["n_estimators"] = 50
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(cfg_dict))

    from ukwind.pipeline import run
    G, P, R = run(load_config(p), synthetic=True)
    assert (tmp_path / "results" / "synthetic_demo" / "summary.md").exists()
    t = P["table"]
    assert t.loc["LightGBM", "MAE"] < t.loc["naive_D-7", "MAE"]
    assert 0 < R["realised"]["energy_mwh"]
    assert G["check"]["model_cf"].between(0.1, 0.7).all()
