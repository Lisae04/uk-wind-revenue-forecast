"""Build the cell and farm tables for the chosen GA portfolio option."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .grid import bng_to_latlon, gridref_to_bng
from .power import choose_iec_class

FARM_COLS = ["option", "farm_id", "mw", "aep_gwh", "cf", "local_authority", "cells"]


def load_portfolio(farms_file: Path, option: int) -> pd.DataFrame:
    if not Path(farms_file).exists():
        raise FileNotFoundError(farms_file)
    df = pd.read_csv(farms_file)
    missing = [c for c in FARM_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"{farms_file} lacks columns {missing}")
    df = df[df["option"] == int(option)].copy()
    if df.empty:
        raise ValueError(f"GA option {option} not found in {farms_file}")
    df["farm_id"] = df["farm_id"].astype(int)
    df["mw"] = pd.to_numeric(df["mw"], errors="raise")
    return df.reset_index(drop=True)


def load_gwa(gwa_file: Path) -> pd.Series:
    """Mean 100 m wind (m/s) per cell_id from the capstone C1 export."""
    if not Path(gwa_file).exists():
        raise FileNotFoundError(
            f"{gwa_file} not found. Copy england_wind.csv from the Capstone Drive folder "
            "into data/raw/ (or run with --synthetic)."
        )
    df = pd.read_csv(gwa_file, usecols=lambda c: c in {"cell_id", "wind_speed_ms"})
    if not {"cell_id", "wind_speed_ms"} <= set(df.columns):
        raise KeyError(f"{gwa_file} must contain cell_id and wind_speed_ms")
    df["wind_speed_ms"] = pd.to_numeric(df["wind_speed_ms"], errors="coerce")
    df = df.dropna().drop_duplicates("cell_id")
    return df.set_index("cell_id")["wind_speed_ms"]


def build_sites(cfg: dict, gwa: pd.Series | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    farms = load_portfolio(cfg["portfolio"]["farms_file"], cfg["portfolio"]["option"])
    if gwa is None:
        gwa = load_gwa(cfg["portfolio"]["gwa_file"])

    rows = []
    for f in farms.itertuples():
        cells = [c.strip() for c in str(f.cells).split(";") if c.strip()]
        for c in cells:
            e, n = gridref_to_bng(c)
            rows.append({"farm_id": f.farm_id, "cell_id": c, "easting": e, "northing": n})
    cells = pd.DataFrame(rows)

    absent = sorted(set(cells["cell_id"]) - set(gwa.index))
    if absent:
        raise KeyError(f"No GWA wind for cells: {absent}")
    cells["gwa_ms"] = cells["cell_id"].map(gwa).astype(float)
    cells["lat"], cells["lon"] = bng_to_latlon(cells["easting"], cells["northing"])
    vave = cfg["turbine"]["iec_class_vave_ms"]
    cells["iec_class"] = [choose_iec_class(v, vave) for v in cells["gwa_ms"]]

    # capacity check: cells x turbine size must equal the GA farm MW
    t = cfg["turbine"]
    cap = cells.groupby("farm_id").size() * t["rated_mw"] * t["turbines_per_cell"]
    bad = (cap - farms.set_index("farm_id")["mw"]).abs() > 1e-6
    if bad.any():
        raise ValueError(f"Cell count x turbine MW != farm MW for farms {list(bad[bad].index)}")

    cent = cells.groupby("farm_id")[["easting", "northing", "lat", "lon", "gwa_ms"]].mean()
    farms = farms.merge(cent, left_on="farm_id", right_index=True)
    farms = farms.rename(columns={"cf": "ga_cf", "aep_gwh": "ga_aep_gwh"})
    farms["name"] = farms["local_authority"].str.replace(" ", "_") + "_" + farms["farm_id"].astype(str)
    return cells, farms[["farm_id", "name", "local_authority", "mw", "ga_aep_gwh", "ga_cf",
                         "easting", "northing", "lat", "lon", "gwa_ms"]]


def summarise(cells: pd.DataFrame, farms: pd.DataFrame) -> str:
    lines = [f"{len(farms)} farms, {len(cells)} cells, {farms['mw'].sum():.0f} MW"]
    for f in farms.itertuples():
        cls = cells.loc[cells.farm_id == f.farm_id, "iec_class"].value_counts().to_dict()
        lines.append(f"  {f.name:<28} {f.mw:5.0f} MW  GWA {f.gwa_ms:4.2f} m/s  "
                     f"({f.lat:.3f}, {f.lon:.3f})  IEC {cls}")
    return "\n".join(lines)
