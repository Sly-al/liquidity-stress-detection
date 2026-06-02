from __future__ import annotations

from sentinel import build_liquidity_sentinel


if __name__ == "__main__":
    result = build_liquidity_sentinel(refresh_sources=False)
    latest = result.aggregation.frame.iloc[-1]
    print(f"Latest date: {latest['date'].date()}")
    print(f"LSI: {latest['lsi']:.1f} ({latest['status']})")
    print(result.commentary)
    print("\nBacktest:")
    print(result.backtest.to_string(index=False))
