"""Explicit live-data smoke check, excluded from deterministic pytest runs."""

from pathlib import Path

from portfolio.analytics import analyze
from portfolio.ingestion import detect_mapping, read_csv, validate
from portfolio.market import fetch_market, market_checks


def main():
    frame = read_csv((Path(__file__).parents[1] / "data/sample.csv").read_bytes())
    txs, errors, _ = validate(frame, detect_mapping(frame.columns))
    assert not errors
    snapshot = fetch_market(tuple(sorted({t.ticker for t in txs} | {"SPY"})), str(txs[0].date))
    errors, warnings = market_checks(txs, snapshot)
    if errors:
        raise RuntimeError("Live market validation failed: " + "; ".join(errors))
    result = analyze(txs, snapshot)
    assert result["metrics"]["market_value"] is not None
    print(
        {
            "status": "live market smoke passed",
            "as_of": result["as_of"],
            "tickers": sorted(snapshot.history),
            "warnings": len(warnings),
            "benchmark_available": not bool(result["benchmark"].get("reason")),
        }
    )


if __name__ == "__main__":
    main()
