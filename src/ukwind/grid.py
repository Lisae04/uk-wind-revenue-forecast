"""Ordnance Survey grid references -> British National Grid -> WGS84."""
from __future__ import annotations

import re

import numpy as np
from pyproj import Transformer

_REF = re.compile(r"^([A-HJ-Z]{2})(\d+)$")
_TO_WGS84 = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)


def gridref_to_bng(ref: str, centre: bool = True) -> tuple[float, float]:
    """'TF8630' -> (easting, northing) in metres.

    Returns the square centre by default (a 4-digit ref is a 1 km square).
    """
    ref = str(ref).strip().upper().replace(" ", "")
    m = _REF.match(ref)
    if not m or len(m.group(2)) % 2:
        raise ValueError(f"Not a valid OS grid reference: {ref!r}")
    letters, digits = m.groups()

    l1, l2 = (ord(c) - ord("A") for c in letters)
    l1 -= l1 > 7          # the grid skips the letter I
    l2 -= l2 > 7
    e100 = ((l1 - 2) % 5) * 5 + (l2 % 5)
    n100 = (19 - (l1 // 5) * 5) - (l2 // 5)

    n = len(digits) // 2
    scale = 10 ** (5 - n)
    e = e100 * 100_000 + int(digits[:n]) * scale
    nn = n100 * 100_000 + int(digits[n:]) * scale
    if centre:
        e, nn = e + scale / 2, nn + scale / 2
    if not (0 <= e <= 700_000 and 0 <= nn <= 1_300_000):
        raise ValueError(f"{ref} falls outside the British National Grid")
    return float(e), float(nn)


def bng_to_latlon(easting, northing):
    lon, lat = _TO_WGS84.transform(np.asarray(easting, float), np.asarray(northing, float))
    return lat, lon
