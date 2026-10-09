"""Physical wind-to-power model.

Chain, applied hour by hour to each 1 km cell:

    1. Shear      U_hub = U_100 * (z_hub / 100) ** alpha_t,
                  alpha_t = ln(U_100 / U_10) / ln(100 / 10)   (clipped to [0, 0.5])
    2. Downscale  U_cell = k_c * U_hub,  k_c = GWA_mean_c / mean_t(U_hub at the NWP point)
    3. Density    U_eq = U_cell * (rho / 1.225) ** (1/3),  rho = p / (R_d * T)
    4. Curve      P_norm = f_IEC(U_eq)   (NREL normalized industry composite)
    5. Output     E_cell = rated_MW * turbines * P_norm * (1 - losses)   [MWh per hour]
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

RHO_STD = 1.225       # kg/m3, IEC standard air density
R_DRY = 287.05        # J/(kg K)
ALPHA_CLIP = (0.0, 0.5)


@lru_cache(maxsize=None)
def load_power_curve(curve_dir: str, iec_class: str) -> tuple[np.ndarray, np.ndarray]:
    num = {"I": 1, "II": 2, "III": 3}[iec_class]
    path = Path(curve_dir) / f"IEC_Class{num}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    ws = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy()
    pw = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy()
    ok = np.isfinite(ws) & np.isfinite(pw)
    ws, pw = ws[ok], np.clip(pw[ok], 0, 1)
    if not np.all(np.diff(ws) > 0):
        raise ValueError(f"Wind speeds in {path} are not increasing")
    return ws, pw


def choose_iec_class(mean_ws: float, vave: dict) -> str:
    """Pick the IEC class whose reference annual mean wind is closest to the site mean."""
    if not np.isfinite(mean_ws):
        raise ValueError("Site mean wind speed is not finite")
    return min(vave, key=lambda c: abs(float(vave[c]) - mean_ws))


def power_norm(ws, curve_dir, iec_class: str) -> np.ndarray:
    """Normalised power (0..1). Zero below cut-in and above cut-out (last curve point)."""
    x, y = load_power_curve(str(curve_dir), iec_class)
    ws = np.nan_to_num(np.asarray(ws, float), nan=0.0)
    return np.interp(ws, x, y, left=0.0, right=0.0)


def shear_alpha(u100, u10) -> np.ndarray:
    u100 = np.asarray(u100, float)
    u10 = np.asarray(u10, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.log(u100 / u10) / np.log(100 / 10)
    a = np.where(np.isfinite(a), a, 1 / 7)
    return np.clip(a, *ALPHA_CLIP)


def to_hub(u100, u10, hub_m: float, ref_m: float = 100.0) -> np.ndarray:
    u100 = np.asarray(u100, float)
    if hub_m == ref_m:
        return u100
    return u100 * (hub_m / ref_m) ** shear_alpha(u100, u10)


def air_density(t2m_c, sp_hpa) -> np.ndarray:
    t = np.asarray(t2m_c, float) + 273.15
    p = np.asarray(sp_hpa, float) * 100.0
    rho = p / (R_DRY * t)
    return np.where(np.isfinite(rho) & (rho > 0.9) & (rho < 1.5), rho, RHO_STD)


def density_equivalent(ws, rho) -> np.ndarray:
    return np.asarray(ws, float) * (np.asarray(rho, float) / RHO_STD) ** (1 / 3)


def cell_energy(u_cell, rho, iec_class, tcfg) -> np.ndarray:
    """MWh produced by one cell in one hour."""
    u = density_equivalent(u_cell, rho) if tcfg.get("air_density_correction", True) else u_cell
    p = power_norm(u, tcfg["power_curve_dir"], iec_class)
    return float(tcfg["rated_mw"]) * int(tcfg["turbines_per_cell"]) * p * (1 - float(tcfg["losses"]))
