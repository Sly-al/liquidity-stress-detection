from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .config import CACHE_DIR, SOURCES, Source
from .utils import business_days


STRESS_EPISODES = [
    ("2014-12-01", "2014-12-31", 1.00, "Декабрь 2014: валютный и денежный стресс"),
    ("2022-02-21", "2022-03-31", 1.25, "Февраль-март 2022: шок ликвидности"),
    ("2023-08-01", "2023-08-31", 0.92, "Август 2023: рост ставок и бюджетный отток"),
]


@dataclass
class DownloadResult:
    key: str
    source: Source
    path: Path
    ok: bool
    message: str


def download_public_sources(timeout: int = 15) -> list[DownloadResult]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    results: list[DownloadResult] = []
    headers = {"User-Agent": "RU-Liquidity-Sentinel/1.0"}
    for key, source in SOURCES.items():
        path = CACHE_DIR / source.cache_name
        try:
            response = requests.get(source.url, timeout=timeout, headers=headers)
            response.raise_for_status()
            path.write_bytes(response.content)
            results.append(DownloadResult(key, source, path, True, f"saved {len(response.content):,} bytes"))
        except Exception as exc:
            if path.exists():
                results.append(DownloadResult(key, source, path, False, f"using cache: {exc}"))
            else:
                results.append(DownloadResult(key, source, path, False, str(exc)))
    return results


def _pulse(dates: pd.DatetimeIndex, start: str, end: str, amplitude: float) -> np.ndarray:
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    mask = (dates >= start_ts) & (dates <= end_ts)
    values = np.zeros(len(dates))
    if mask.any():
        span = max(mask.sum() - 1, 1)
        phase = np.linspace(-2.2, 2.2, mask.sum())
        values[mask] = amplitude * np.exp(-(phase**2) / 1.55)
        after = (dates > end_ts) & (dates <= end_ts + pd.Timedelta(days=35))
        values[after] = amplitude * 0.28 * np.exp(-np.linspace(0, 3, after.sum()))
    return values


def _tax_flags(dates: pd.DatetimeIndex) -> pd.DataFrame:
    frame = pd.DataFrame({"date": dates})
    day = frame["date"].dt.day
    frame["tax_due_core"] = day.isin([25, 28]).astype(int)
    frame["tax_week_flag"] = day.between(21, 29).astype(int)
    frame["end_of_month_flag"] = (frame["date"].dt.is_month_end | (day >= 27)).astype(int)
    frame["end_of_quarter_flag"] = (
        frame["date"].dt.month.isin([3, 6, 9, 12]) & frame["end_of_month_flag"].eq(1)
    ).astype(int)
    frame["seasonal_factor"] = (
        1.0
        + 0.22 * frame["tax_week_flag"]
        + 0.08 * frame["end_of_month_flag"]
        + 0.10 * frame["end_of_quarter_flag"]
    ).clip(1.0, 1.4)
    return frame


def generate_market_frame(start: str = "2013-01-01", end: str | None = None, seed: int = 17) -> pd.DataFrame:
    dates = business_days(start, end)
    rng = np.random.default_rng(seed)
    t = np.arange(len(dates))
    frame = _tax_flags(dates)

    stress = np.zeros(len(dates))
    for start_date, end_date, amplitude, _ in STRESS_EPISODES:
        stress += _pulse(dates, start_date, end_date, amplitude)
    stress += _pulse(dates, "2026-02-01", "2026-03-15", 0.45)
    stress += 0.10 * frame["tax_week_flag"].to_numpy() + 0.06 * frame["end_of_quarter_flag"].to_numpy()
    stress = np.clip(stress, 0, None)
    frame["latent_stress"] = stress

    cycle = np.sin(t / 65.0) + 0.45 * np.sin(t / 250.0)
    noise = rng.normal(0, 1, len(dates))
    repo_days = frame["date"].dt.weekday.isin([1, 3]).astype(int).to_numpy()
    ofz_days = frame["date"].dt.weekday.isin([2]).astype(int).to_numpy()

    frame["m1_actual_corr_accounts"] = 1900 + 130 * cycle + 480 * stress + rng.normal(0, 55, len(dates))
    frame["m1_required_reserves"] = 1700 + 65 * np.sin(t / 310.0) + rng.normal(0, 18, len(dates))
    frame["m1_accounting_reserves"] = 360 + 20 * np.sin(t / 120.0) + rng.normal(0, 8, len(dates))
    frame["m1_ruonia"] = 7.2 + 1.6 * np.sin(t / 420.0) + 5.1 * stress + 0.25 * noise
    frame["m1_end_period_flag"] = frame["end_of_month_flag"]

    frame["m2_term_days"] = np.where(repo_days == 1, 7, np.nan)
    frame["m2_repo_demand"] = np.where(repo_days == 1, 580 + 920 * stress + rng.normal(0, 75, len(dates)), np.nan)
    frame["m2_repo_allotment"] = np.where(repo_days == 1, 470 + 150 * np.sin(t / 80) + rng.normal(0, 55, len(dates)), np.nan)
    frame["m2_key_rate"] = 7.5 + 1.2 * np.sin(t / 500.0) + 4.2 * stress
    frame["m2_cutoff_rate"] = frame["m2_key_rate"] + np.where(repo_days == 1, 0.22 + 1.15 * stress + rng.normal(0, 0.08, len(dates)), np.nan)

    frame["m3_offer"] = np.where(ofz_days == 1, 105 + 18 * np.sin(t / 55) + rng.normal(0, 8, len(dates)), np.nan)
    frame["m3_demand"] = np.where(ofz_days == 1, frame["m3_offer"] * (2.1 - 0.95 * stress + rng.normal(0, 0.13, len(dates))), np.nan)
    frame["m3_placed"] = np.where(ofz_days == 1, np.minimum(frame["m3_offer"], frame["m3_demand"] * 0.92), np.nan)
    frame["m3_weighted_yield"] = np.where(ofz_days == 1, 8.0 + 1.4 * np.sin(t / 360) + 2.8 * stress + rng.normal(0, 0.12, len(dates)), np.nan)
    frame["m3_curve_benchmark"] = np.where(ofz_days == 1, 7.9 + 1.2 * np.sin(t / 360), np.nan)

    treasury_base = 2100 + 280 * np.sin(t / 170.0)
    drain = 460 * stress + 180 * frame["tax_week_flag"].to_numpy()
    frame["m5_bank_treasury_balances"] = treasury_base - drain + rng.normal(0, 65, len(dates))
    frame["m5_eks_deposits"] = 850 + 145 * np.sin(t / 90.0) - 280 * stress + rng.normal(0, 45, len(dates))
    frame["m5_bank_count"] = (24 + 5 * np.sin(t / 100.0) - 5 * stress + rng.normal(0, 1.5, len(dates))).clip(8, 45)

    frame["ground_truth_liquidity_stress"] = (15 + 72 * stress + 8 * frame["tax_week_flag"] + rng.normal(0, 4, len(dates))).clip(0, 100)
    return frame
