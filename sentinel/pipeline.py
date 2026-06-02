from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .aggregation import AggregationResult, aggregate_lsi, combine_modules, sensitivity_analysis
from .backtest import run_backtest
from .data import DownloadResult, download_public_sources, generate_market_frame
from .llm import automatic_commentary
from .modules import build_module_outputs


@dataclass
class SentinelResult:
    raw: pd.DataFrame
    modules: dict[str, pd.DataFrame]
    aggregation: AggregationResult
    backtest: pd.DataFrame
    sensitivity: pd.DataFrame
    source_results: list[DownloadResult]
    commentary: str


def build_liquidity_sentinel(refresh_sources: bool = False) -> SentinelResult:
    source_results = download_public_sources() if refresh_sources else []
    raw = generate_market_frame(end=pd.Timestamp.today().strftime("%Y-%m-%d"))
    modules = build_module_outputs(raw)
    combined = combine_modules(raw, modules)
    aggregation = aggregate_lsi(combined)
    backtest = run_backtest(aggregation.frame)
    sensitivity = sensitivity_analysis(aggregation)
    commentary = automatic_commentary(aggregation.frame.iloc[-1])
    return SentinelResult(raw, modules, aggregation, backtest, sensitivity, source_results, commentary)
