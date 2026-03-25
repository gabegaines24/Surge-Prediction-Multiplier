"""
Shared calendar / time features for training and API inference.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# US federal (observed) + common NYC closures in early 2025 (date-normalized)
_HOLIDAY_DATES = pd.to_datetime(
    [
        "2025-01-01",
        "2025-01-20",
        "2025-02-17",
        "2025-12-25",  # future-proof list for rolling years
    ]
)


def nyc_holiday_mask(dates: pd.Series) -> pd.Series:
    """Boolean Series: True if calendar date is in holiday set."""
    norm = pd.to_datetime(dates).dt.normalize()
    return norm.isin(_HOLIDAY_DATES.normalize())


def add_calendar_and_rush_features(df: pd.DataFrame, time_col: str = "Time_Bin") -> pd.DataFrame:
    """Add month cyclical, holiday flag, rush-hour flag (uses Time_Bin)."""
    out = df.copy()
    ts = out[time_col]
    out["month"] = ts.dt.month
    out["month_sin"] = np.sin(2 * np.pi * (out["month"] - 1) / 12)
    out["month_cos"] = np.cos(2 * np.pi * (out["month"] - 1) / 12)
    hour = ts.dt.hour
    out["is_rush_hour"] = (hour.between(7, 9) | hour.between(17, 19)).astype(int)
    out["is_holiday"] = nyc_holiday_mask(ts).astype(int)
    return out


def add_calendar_from_hour_dow(
    hour: int,
    day_of_week: int,
    month: int | None = None,
    is_holiday: int | None = None,
) -> dict:
    """
    Build calendar feature dict for API when only hour + DOW known.
    If month is None, defaults to 1 (January) for cyclical encoding.
    If is_holiday is None, defaults to 0.
    """
    m = int(month) if month is not None else 1
    m = max(1, min(12, m))
    hol = int(is_holiday) if is_holiday is not None else 0
    hol = 1 if hol else 0
    rush = 1 if (7 <= hour <= 9) or (17 <= hour <= 19) else 0
    return {
        "month": m,
        "month_sin": float(np.sin(2 * np.pi * (m - 1) / 12)),
        "month_cos": float(np.cos(2 * np.pi * (m - 1) / 12)),
        "is_rush_hour": rush,
        "is_holiday": hol,
    }
