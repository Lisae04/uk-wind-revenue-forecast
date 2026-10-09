"""Load and validate config.yaml."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]

REQUIRED = {
    "period": ["start", "end", "test_start"],
    "market": ["tz", "gate_closure_local", "price_provider"],
    "portfolio": ["farms_file", "option", "gwa_file"],
    "turbine": ["rated_mw", "turbines_per_cell", "hub_height_m", "gwa_height_m",
                "losses", "iec_class_vave_ms", "power_curve_dir"],
    "reference_farms": ["mode", "bmus"],
    "generation_model": ["retrain_every_days", "lgbm"],
    "price_model": ["lear_window_days", "lear_recalibrate_every_days",
                    "lgbm_retrain_every_days", "lgbm"],
    "paths": ["cache", "processed", "results"],
}


def load_config(path: str | Path = ROOT / "config.yaml") -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f)

    missing = [f"{sec}.{k}" for sec, keys in REQUIRED.items()
               for k in keys if k not in (cfg.get(sec) or {})]
    if missing:
        raise KeyError(f"config.yaml is missing: {', '.join(missing)}")

    p = cfg["period"]
    start, end, test = (pd.Timestamp(p[k]) for k in ("start", "end", "test_start"))
    if not start < test < end:
        raise ValueError("period must satisfy start < test_start < end")

    t = cfg["turbine"]
    if not 0 <= float(t["losses"]) < 1:
        raise ValueError("turbine.losses must be in [0, 1)")
    if cfg["reference_farms"]["mode"] not in {"auto", "bmu", "national"}:
        raise ValueError("reference_farms.mode must be auto, bmu or national")
    for b in cfg["reference_farms"]["bmus"] or []:
        for k in ("bmu", "lat", "lon", "capacity_mw"):
            if k not in b:
                raise KeyError(f"reference_farms.bmus entry {b} lacks '{k}'")

    # resolve relative paths against the repo root
    for k in ("cache", "processed", "results"):
        cfg["paths"][k] = ROOT / cfg["paths"][k]
        cfg["paths"][k].mkdir(parents=True, exist_ok=True)
    for k in ("farms_file", "gwa_file"):
        cfg["portfolio"][k] = ROOT / cfg["portfolio"][k]
    cfg["turbine"]["power_curve_dir"] = ROOT / cfg["turbine"]["power_curve_dir"]
    return cfg


def reference_mode(cfg: dict) -> str:
    mode = cfg["reference_farms"]["mode"]
    if mode == "auto":
        return "bmu" if cfg["reference_farms"]["bmus"] else "national"
    if mode == "bmu" and not cfg["reference_farms"]["bmus"]:
        raise ValueError("reference_farms.mode is 'bmu' but no bmus are listed")
    return mode
