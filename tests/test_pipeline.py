from __future__ import annotations

from sentinel import build_liquidity_sentinel


def test_pipeline_builds_lsi_and_backtest():
    result = build_liquidity_sentinel(refresh_sources=False)
    frame = result.aggregation.frame
    assert {"lsi", "status", "m1_contribution", "m5_contribution"}.issubset(frame.columns)
    assert frame["lsi"].between(0, 100).all()
    assert len(result.backtest) == 3
    assert result.backtest["detected"].all()
