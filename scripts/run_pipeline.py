"""Run the full pipeline.

    python scripts/run_pipeline.py              # real data (Open-Meteo + Elexon, no keys)
    python scripts/run_pipeline.py --synthetic  # offline demo on synthetic data
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ukwind.config import load_config  # noqa: E402
from ukwind.pipeline import run  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--synthetic", action="store_true", help="use synthetic data (no network)")
    a = ap.parse_args()
    cfg = load_config(a.config) if a.config else load_config()
    run(cfg, synthetic=a.synthetic)
