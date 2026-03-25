"""
Lightweight TLC-style zone hints (airport / Manhattan proxy).
Extend via config or data file as needed.
"""
from __future__ import annotations

import pandas as pd

# Common NYC TLC location IDs (approximate; verify against your zone map)
JFK_ZONES = {132}
LGA_ZONES = {138, 244}
EWR_ZONE = {1}
MANHATTAN_SOUTH = set(range(4, 13))  # Financial / downtown core (illustrative subset)
MANHATTAN_MID = set(range(43, 82))

AIRPORT_ZONES = JFK_ZONES | LGA_ZONES | EWR_ZONE
# Broader Manhattan core (zones used heavily in TLC data — illustrative)
MANHATTAN_CORE = MANHATTAN_SOUTH | MANHATTAN_MID | {161, 162, 163, 164, 170, 186, 234, 237}


def add_zone_hint_features(df: pd.DataFrame, zone_col: str = "Zone") -> pd.DataFrame:
    """Add binary flags from zone ID (vectorized)."""
    out = df.copy()
    z = out[zone_col].astype(int)
    out["is_airport_zone"] = z.isin(AIRPORT_ZONES).astype(int)
    out["is_manhattan_core"] = z.isin(MANHATTAN_CORE).astype(int)
    return out
