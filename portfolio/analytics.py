"""Portfolio metrics, historical valuation, and same-flow price benchmark."""

import math
from datetime import date
from decimal import Decimal

import pandas as pd

from portfolio.holdings import reconstruct
from portfolio.market import split_events
from portfolio.xirr import xirr

D = Decimal


def analyze(
    transactions,
    snapshot,
    benchmark="SPY",
    overrides=None,
    threshold=0.10,
    effective_limit=10,
):
    overrides = overrides or {}
    splits = split_events(snapshot)
    positions, cashflows, notices = reconstruct(transactions, splits)
    held = [ticker for ticker, p in positions.items() if p.shares]
    dates = [
        snapshot.history[t].index[-1].date()
        for t in held
        if t in snapshot.history and not snapshot.history[t].empty
    ]
    # Common valuation date prevents mixing returns from different closes.
    asof = min(dates) if dates else max(t.date for t in transactions)
    if held and asof < max(t.date for t in transactions):
        raise ValueError(
            "A trade is newer than the available market close. Refresh after market close."
        )
    rows, missing = [], []
    exact_values = []
    day_change = D(0)
    day_change_valid = True
    for ticker, pos in positions.items():
        frame = snapshot.history.get(ticker, pd.DataFrame())
        available = frame.loc[frame.index <= pd.Timestamp(asof)] if not frame.empty else frame
        price = None
        source = "Yahoo close"
        price_date = None
        if ticker in overrides:
            price = D(str(overrides[ticker]))
            source = "user-entered"
            price_date = asof
        elif not available.empty:
            price = D(str(float(available.Close.iloc[-1])))
            price_date = available.index[-1].date()
        if pos.shares and (price is None or price_date != asof):
            missing.append(ticker)
        value = pos.shares * price if price is not None else (D(0) if not pos.shares else None)
        gain = value - pos.basis if value is not None else None
        if value is not None:
            exact_values.append(value)
        if pos.shares:
            if len(available) >= 2 and source != "user-entered":
                day_change += pos.shares * D(
                    str(float(available.Close.iloc[-1] - available.Close.iloc[-2]))
                )
            else:
                day_change_valid = False
        rows.append(
            {
                "ticker": ticker,
                "shares": float(pos.shares),
                "cost_basis": float(pos.basis),
                "avg_cost": float(pos.basis / pos.shares) if pos.shares else 0,
                "price": float(price) if price is not None else None,
                "market_value": float(value) if value is not None else None,
                "unrealized_gain": float(gain) if gain is not None else None,
                "unrealized_pct": float(gain / pos.basis * 100)
                if gain is not None and pos.basis
                else None,
                "realized_gain": float(pos.realized),
                "price_source": source,
                "price_date": str(price_date or "unavailable"),
                "sector": snapshot.metadata.get(ticker, {}).get("sector", "Unclassified"),
            }
        )
    invested = sum((t.quantity * t.price + t.fees for t in transactions if t.action == "BUY"), D(0))
    proceeds = sum(
        (t.quantity * t.price - t.fees for t in transactions if t.action == "SELL"),
        D(0),
    )
    basis = sum((p.basis for p in positions.values()), D(0))
    realized = sum((p.realized for p in positions.values()), D(0))
    value = None if missing else sum(exact_values, D(0))
    gain = None if value is None else value + proceeds - invested
    if value is not None and abs(gain - (value - basis + realized)) > D("0.005"):
        raise ValueError("Accounting identity failed; analysis stopped.")
    terminal_flows = cashflows + ([(asof, value)] if held and value is not None else [])
    irr = xirr(terminal_flows) if not missing else None
    for row in rows:
        row["weight_pct"] = row["market_value"] / float(value) * 100 if value else None
    weights = [r["weight_pct"] / 100 for r in rows if r["shares"] and r["weight_pct"] is not None]
    hhi = sum(w * w for w in weights)
    effective = 1 / hhi if hhi else None
    flags = [
        f"{r['ticker']} exceeds {threshold:.0%} by holding."
        for r in rows
        if r["weight_pct"] is not None and r["weight_pct"] / 100 > threshold
    ]
    if effective and effective < effective_limit:
        flags.append(f"Effective holdings {effective:.2f} are below {effective_limit}.")
    if sum(sorted(weights, reverse=True)[:3]) > 0.6:
        flags.append("Top three holdings exceed 60% of market value.")
    history, bench = time_series(transactions, snapshot, cashflows, asof, benchmark)
    if overrides:
        notices.append(
            "User-entered prices affect current metrics only; historical charts retain market closes."
        )
    return {
        "as_of": str(asof),
        "fetched_at": snapshot.fetched_at,
        "holdings": rows,
        "metrics": {
            "market_value": float(value) if value is not None else None,
            "cost_basis": float(basis),
            "total_invested": float(invested),
            "sale_proceeds": float(proceeds),
            "net_invested": float(invested - proceeds),
            "realized_gain": float(realized),
            "unrealized_gain": float(value - basis) if value is not None else None,
            "total_gain": float(gain) if gain is not None else None,
            "cumulative_return_pct": float(gain / invested * 100)
            if gain is not None and invested
            else None,
            "xirr_pct": irr.rate * 100 if irr and irr.rate is not None else None,
            "day_change": float(day_change) if day_change_valid and not missing else None,
        },
        "xirr_reason": irr.reason if irr else "Missing valuation prices.",
        "history_days": (asof - transactions[0].date).days,
        "cashflows": [
            {"date": str(day), "amount": float(amount)} for day, amount in terminal_flows
        ],
        "concentration": {"hhi": hhi, "effective_holdings": effective, "flags": flags},
        "benchmark": bench,
        "history": history,
        "notices": notices,
        "missing": missing,
        "stale": (date.today() - asof).days > 4,
        "valuation_sources": sorted({r["price_source"] for r in rows if r["shares"]}),
    }


def time_series(transactions, snapshot, cashflows, asof, benchmark):
    dates = pd.date_range(transactions[0].date, asof, freq="D")
    if dates.empty:
        return pd.DataFrame(), {"ticker": benchmark, "reason": "No history."}
    splits = split_events(snapshot)
    tickers = sorted({t.ticker for t in transactions})
    shares = pd.DataFrame(0.0, index=dates, columns=tickers)
    buy_flows = pd.Series(0.0, index=dates)
    sale_flows = buy_flows.copy()
    for tx in transactions:
        factor = math.prod(ratio for day, ratio in splits.get(tx.ticker, []) if day > tx.date)
        day = pd.Timestamp(tx.date)
        if day not in dates:
            continue
        shares.loc[day, tx.ticker] += (
            float(tx.quantity) * factor * (1 if tx.action == "BUY" else -1)
        )
        if tx.action == "BUY":
            buy_flows.loc[day] += float(tx.quantity * tx.price + tx.fees)
        else:
            sale_flows.loc[day] += float(tx.quantity * tx.price - tx.fees)
    shares = shares.cumsum()
    values = pd.Series(0.0, index=dates)
    valid = pd.Series(True, index=dates)
    for ticker in tickers:
        frame = snapshot.history.get(ticker, pd.DataFrame())
        prices = (
            frame.Close.reindex(dates).ffill()
            if not frame.empty
            else pd.Series(float("nan"), index=dates)
        )
        # Before first available close (weekend trade) use the next available close only.
        if not frame.empty:
            next_close = frame.Close.reindex(dates).bfill()
            prices = prices.fillna(next_close)
        active = shares[ticker].abs() > 1e-9
        valid &= ~active | prices.notna()
        values += (shares[ticker] * prices).where(active, 0).fillna(0)
    values = values.where(valid)
    invested, proceeds = buy_flows.cumsum(), sale_flows.cumsum()
    history = pd.DataFrame(
        {
            "portfolio_value": values,
            "net_invested": invested - proceeds,
            "total_invested": invested,
            "sale_proceeds": proceeds,
        }
    )
    history["cumulative_return_pct"] = (
        (values + proceeds - invested) / invested.replace(0, float("nan")) * 100
    )
    frame = snapshot.history.get(benchmark, pd.DataFrame())
    bench = {
        "ticker": benchmark,
        "basis": "price return, same cash flows",
        "reason": "",
    }
    if frame.empty or benchmark in snapshot.errors:
        bench["reason"] = "Benchmark data unavailable or currency unverified."
        return history, bench
    units = 0.0
    flows = []
    for day, amount in cashflows:
        available = frame.loc[frame.index >= pd.Timestamp(day)]
        if available.empty:
            bench["reason"] = "Missing benchmark close for a cash-flow date."
            return history, bench
        units -= float(amount) / float(available.Close.iloc[0])
        flows.append((day, amount))
        if units < -1e-8:
            bench["reason"] = "Sales exceed same-flow benchmark value; comparison unavailable."
            return history, bench
    terminal = frame.loc[frame.index <= pd.Timestamp(asof)]
    if terminal.empty or terminal.index[-1].date() != asof:
        bench["reason"] = "No benchmark price at the portfolio valuation close."
        return history, bench
    value = units * float(terminal.Close.iloc[-1])
    total_invested = sum(-float(v) for _, v in cashflows if v < 0)
    sales = sum(float(v) for _, v in cashflows if v > 0)
    gain = value + sales - total_invested
    irr = xirr(flows + [(asof, value)])
    bench.update(
        market_value=value,
        total_gain=gain,
        cumulative_return_pct=gain / total_invested * 100 if total_invested else None,
        xirr_pct=irr.rate * 100 if irr.rate is not None else None,
        as_of=str(asof),
        eligible=(asof - transactions[0].date).days >= 90,
    )
    return history, bench
