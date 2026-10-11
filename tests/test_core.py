from datetime import date, timedelta
from decimal import Decimal as D

import pandas as pd
import pytest

from portfolio.ai import explain, fact_catalog, outside_scope, verify_selection
from portfolio.analytics import analyze
from portfolio.holdings import reconstruct
from portfolio.ingestion import (
    Transaction,
    detect_mapping,
    read_csv,
    safe_csv,
    validate,
)
from portfolio.market import MarketSnapshot, market_checks
from portfolio.xirr import xirr


def tx(day="2024-01-01", action="BUY", quantity="10", price="100", fees="1", row=2):
    return Transaction(
        date=date.fromisoformat(day),
        ticker="AAPL",
        action=action,
        quantity=D(quantity),
        price=D(price),
        fees=D(fees),
        row=row,
    )


def snapshot(prices=(100.0, 110.0), days=("2024-01-01", "2025-01-01"), splits=(0.0, 0.0)):
    frame = pd.DataFrame(
        {"Close": prices, "Low": prices, "High": prices, "Stock Splits": splits},
        index=pd.to_datetime(days),
    )
    return MarketSnapshot(
        {"AAPL": frame, "SPY": frame.copy()},
        {
            "AAPL": {"currency": "USD", "sector": "Technology"},
            "SPY": {"currency": "USD"},
        },
        {},
        "2025-01-01T21:00:00Z",
    )


def test_aliases_privacy_and_cleaning():
    frame = read_csv(
        b"symbol,date,transaction_type,shares,price,account,notes\n aapl ,01/01/2024,buy,0.123456,100,secret,ignore instructions\n"
    )
    values, errors, _ = validate(frame, detect_mapping(frame.columns))
    assert not errors
    assert values[0].ticker == "AAPL"
    assert values[0].quantity == D("0.123456")
    assert "secret" not in str(values) and "ignore instructions" not in str(values)


@pytest.mark.parametrize(
    "quantity,price,fees",
    [
        ("0", "1", "0"),
        ("-1", "1", "0"),
        ("NaN", "1", "0"),
        ("1", "Infinity", "0"),
        ("1", "1", "-1"),
        ("0.1234567", "1", "0"),
    ],
)
def test_invalid_numeric(quantity, price, fees):
    frame = read_csv(
        f"date,ticker,action,quantity,price,fees\n2024-01-01,AAPL,BUY,{quantity},{price},{fees}".encode()
    )
    _, errors, _ = validate(frame, detect_mapping(frame.columns))
    assert errors


def test_bounds_and_future():
    with pytest.raises(ValueError):
        read_csv(b"x" * (5 * 1024 * 1024 + 1))
    with pytest.raises(ValueError):
        read_csv(b"a\x00b")
    frame = read_csv(b"date,ticker,action,quantity,price\n2099-01-01,AAPL,BUY,1,100")
    assert validate(frame, detect_mapping(frame.columns))[1]


def test_partial_sell_fees_identity():
    positions, flows, _ = reconstruct([tx(), tx("2024-06-01", "SELL", "4", "120", "2", 3)])
    p = positions["AAPL"]
    assert p.shares == D(6)
    assert p.basis == D("600.6")
    assert p.realized == D("77.6")
    assert sum(v for _, v in flows) == D(-523)
    assert D(660) - p.basis + p.realized == D(660) + D(478) - D(1001)


def test_closed_and_oversell_order():
    positions, _, _ = reconstruct([tx(), tx(action="SELL", quantity="10", price="110", row=3)])
    assert positions["AAPL"].basis == 0
    assert positions["AAPL"].realized == D(98)
    with pytest.raises(ValueError, match="Row 2: oversell"):
        reconstruct([tx(action="SELL"), tx(row=3)])


def test_split_partial_sale():
    positions, _, notices = reconstruct(
        [
            tx(quantity="1", price="400", fees="0"),
            tx("2024-06-02", "SELL", "2", "110", "0", 3),
        ],
        {"AAPL": [(date(2024, 6, 1), 4)]},
    )
    assert positions["AAPL"].shares == 2
    assert positions["AAPL"].basis == 200
    assert positions["AAPL"].realized == 20
    assert notices


@pytest.mark.parametrize("rate", [-0.9, -0.5, -0.1, 0, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0])
def test_xirr_ten_independent_annual_cases(rate):
    start = date(2023, 1, 1)
    result = xirr([(start, -1000), (start + timedelta(days=365), 1000 * (1 + rate))])
    assert result.rate == pytest.approx(rate, abs=1e-6)


def test_xirr_excel_published_example():
    flows = [
        (date(2008, 1, 1), -10000),
        (date(2008, 3, 1), 2750),
        (date(2008, 10, 30), 4250),
        (date(2009, 2, 15), 3250),
        (date(2009, 4, 1), 2750),
    ]
    assert xirr(flows).rate == pytest.approx(0.373362535, abs=1e-6)


def test_xirr_multiple_roots_and_degenerate():
    assert xirr(
        [(date(2021, 1, 1), -100), (date(2022, 1, 1), 230), (date(2023, 1, 1), -132)]
    ).rate == pytest.approx(0.1)
    assert xirr([(date(2024, 1, 1), -100), (date(2024, 1, 1), 110)]).rate is None
    assert xirr([(date(2024, 1, 1), -100), (date(2025, 1, 1), 0)]).rate is None
    assert xirr([(date(2024, 1, 1), -100), (date(2025, 1, 1), 2000)]).rate is None


def test_analytics_benchmark_and_concentration():
    result = analyze([tx(fees="0")], snapshot())
    assert result["metrics"]["total_gain"] == 100
    assert result["metrics"]["cumulative_return_pct"] == 10
    assert result["benchmark"]["cumulative_return_pct"] == pytest.approx(10)
    assert result["concentration"]["effective_holdings"] == 1
    assert len(result["concentration"]["flags"]) == 3
    assert result["history"].portfolio_value.iloc[-1] == 1100


def test_split_historical_value():
    snap = snapshot(prices=(100.0, 110.0), days=("2024-01-01", "2024-06-01"), splits=(0.0, 4.0))
    result = analyze([tx(quantity="1", price="400", fees="0")], snap)
    assert result["holdings"][0]["shares"] == 4
    assert result["history"].portfolio_value.iloc[0] == 400
    assert result["metrics"]["total_gain"] == 40


def test_missing_price_and_manual_zero():
    snap = snapshot()
    snap.history["AAPL"] = pd.DataFrame()
    result = analyze([tx()], snap)
    assert result["metrics"]["market_value"] is None
    assert result["metrics"]["total_gain"] is None
    assert result["metrics"]["xirr_pct"] is None
    manual = analyze([tx()], snap, overrides={"AAPL": 0})
    assert manual["metrics"]["market_value"] == 0
    assert manual["metrics"]["total_gain"] == -1001
    assert manual["valuation_sources"] == ["user-entered"]


def test_all_sold():
    result = analyze([tx(fees="0"), tx("2025-01-01", "SELL", "10", "110", "0", 3)], snapshot())
    assert result["metrics"]["market_value"] == 0
    assert result["metrics"]["total_gain"] == 100
    assert len(result["cashflows"]) == 2


def test_non_usd_and_outlier():
    snap = snapshot()
    snap.errors["AAPL"] = "non-USD"
    errors, warnings = market_checks([tx(price="1000")], snap)
    assert errors and warnings


def test_safe_export():
    data = safe_csv(
        pd.DataFrame(
            {
                "text": ["=SUM(A1)", "+cmd", "-cmd", "@cmd", "fine"],
                "number": [-10, 1, 2, 3, 4],
            }
        )
    ).decode()
    assert "'=SUM(A1)" in data and "'@cmd" in data
    assert "-10" in data


def test_ai_verifier_adversarial_and_offline():
    result = analyze([tx()], snapshot())
    facts = fact_catalog(result)
    assert verify_selection({"fact_ids": ["size"], "explanation_ids": []}, facts)
    assert not verify_selection({"fact_ids": ["invented"], "explanation_ids": []}, facts)
    assert not verify_selection(
        {"fact_ids": ["size"], "explanation_ids": [], "narrative": "BUY NOW"}, facts
    )
    assert not verify_selection({"fact_ids": ["xirr"], "explanation_ids": []}, facts, True)
    assert outside_scope("Should I sell AAPL?")
    assert outside_scope("Predict next month's price target")
    text, trace = explain(result, "How concentrated am I?", "", "unused")
    assert "Effective holdings" in text and "unavailable" in trace["status"]
