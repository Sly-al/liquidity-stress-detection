from __future__ import annotations

import pandas as pd

from .data import STRESS_EPISODES


def run_backtest(lsi_frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for start, end, _, label in STRESS_EPISODES:
        mask = lsi_frame["date"].between(pd.Timestamp(start), pd.Timestamp(end))
        before = lsi_frame["date"].between(pd.Timestamp(start) - pd.Timedelta(days=21), pd.Timestamp(start) - pd.Timedelta(days=1))
        episode = lsi_frame.loc[mask]
        prior = lsi_frame.loc[before]
        rows.append(
            {
                "episode": label,
                "period": f"{start} - {end}",
                "avg_lsi": round(float(episode["lsi"].mean()), 1),
                "max_lsi": round(float(episode["lsi"].max()), 1),
                "avg_prior_lsi": round(float(prior["lsi"].mean()), 1) if not prior.empty else None,
                "lead_warning_days": _lead_warning_days(lsi_frame, start),
                "detected": bool((episode["lsi"] >= 70).any()),
            }
        )
    return pd.DataFrame(rows)


def _lead_warning_days(lsi_frame: pd.DataFrame, start: str) -> int:
    start_ts = pd.Timestamp(start)
    window = lsi_frame[lsi_frame["date"].between(start_ts - pd.Timedelta(days=14), start_ts)]
    alerts = window[window["lsi"] >= 40]
    if alerts.empty:
        return 0
    return int((start_ts - alerts["date"].min()).days)
