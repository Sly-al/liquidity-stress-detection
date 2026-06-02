from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_mad_score(series: pd.Series, window: int = 756, min_periods: int = 60) -> pd.Series:
    """Robust rolling z-score using median absolute deviation."""
    median = series.rolling(window, min_periods=min_periods).median()

    def mad(values: np.ndarray) -> float:
        med = np.nanmedian(values)
        value = np.nanmedian(np.abs(values - med))
        return value if value > 1e-9 else np.nan

    dispersion = series.rolling(window, min_periods=min_periods).apply(mad, raw=True)
    score = 0.6745 * (series - median) / dispersion
    return score.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-8, 8)


def sigmoid(x: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def stress_status(value: float) -> str:
    if value >= 70:
        return "КРАСНЫЙ"
    if value >= 40:
        return "ЖЁЛТЫЙ"
    return "ЗЕЛЁНЫЙ"


def stress_color(value: float) -> str:
    if value >= 70:
        return "#c81e1e"
    if value >= 40:
        return "#b7791f"
    return "#1f8a4c"


def business_days(start: str = "2013-01-01", end: str | None = None) -> pd.DatetimeIndex:
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    return pd.bdate_range(start=start, end=end)
