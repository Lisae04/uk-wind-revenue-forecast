"""List Elexon wind BM units that report metered output (B1610), to pick reference farms.

Most English onshore farms are embedded in the distribution network and have no BM unit,
so expect the list to be dominated by Scottish onshore and offshore farms. Check each
candidate's location in the UK Renewable Energy Planning Database (REPD) before adding it
to config.yaml under reference_farms.bmus with lat, lon and capacity_mw.

    python scripts/find_wind_bmus.py            # all wind BM units
    python scripts/find_wind_bmus.py --check    # also test that B1610 data exists (slower)
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ukwind.config import load_config  # noqa: E402
from ukwind.fetch_elexon import fetch_bmu_output, fetch_bmu_reference  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    ref = fetch_bmu_reference(cfg["paths"]["cache"])
    need = {"elexonBmUnit", "fuelType", "bmUnitName", "generationCapacity", "gspGroupName"}
    if not need <= set(ref.columns):
        raise SystemExit(f"Unexpected reference schema: {sorted(ref.columns)}")
    wind = ref[ref["fuelType"].astype(str).str.upper() == "WIND"].copy()
    wind["generationCapacity"] = pd.to_numeric(wind["generationCapacity"], errors="coerce")
    wind = wind.sort_values(["gspGroupName", "generationCapacity"], ascending=[True, False])
    cols = ["elexonBmUnit", "nationalGridBmUnit", "bmUnitName", "leadPartyName",
            "generationCapacity", "gspGroupName"]
    if a.check:
        ok = []
        for b in wind["elexonBmUnit"]:
            try:
                s = fetch_bmu_output(b, "2025-06-01", "2025-06-07", cfg["paths"]["cache"])
                ok.append(int(s.notna().sum()))
            except Exception:
                ok.append(0)
        wind["hours_with_data_jun25"] = ok
        cols.append("hours_with_data_jun25")
    out = cfg["paths"]["processed"] / "wind_bmus.csv"
    wind[cols].to_csv(out, index=False)
    with pd.option_context("display.max_rows", 400, "display.width", 160):
        print(wind[cols].to_string(index=False))
    print(f"\n{len(wind)} wind BM units -> {out}")
