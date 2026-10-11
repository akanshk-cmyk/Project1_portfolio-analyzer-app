"""Average cost using Decimal. Split events change shares, never cost basis."""

from dataclasses import dataclass
from decimal import Decimal

D = Decimal


@dataclass
class Position:
    shares: Decimal = D(0)
    basis: Decimal = D(0)
    realized: Decimal = D(0)


def reconstruct(transactions, splits=None, through=None):
    positions, cashflows, notices = {}, [], []
    events = []
    for tx in transactions:
        if through is None or tx.date <= through:
            events.append((tx.date, 1, tx.time, tx.row, tx))
    for ticker, actions in (splits or {}).items():
        for day, ratio in actions:
            if through is None or day <= through:
                events.append((day, 0, "", 0, (ticker, D(str(ratio)))))
    for day, kind, _, _, item in sorted(events, key=lambda e: e[:4]):
        if kind == 0:
            ticker, ratio = item
            if not ratio.is_finite() or ratio <= 0:
                raise ValueError(f"Invalid split ratio for {ticker} on {day}")
            pos = positions.get(ticker)
            if pos and pos.shares:
                pos.shares *= ratio
                notices.append(f"{ticker}: adjusted for {ratio}:1 split on {day}.")
            continue
        tx = item
        pos = positions.setdefault(tx.ticker, Position())
        if tx.action == "BUY":
            cost = tx.quantity * tx.price + tx.fees
            pos.shares += tx.quantity
            pos.basis += cost
            cashflows.append((tx.date, -cost))
        else:
            if tx.quantity > pos.shares:
                raise ValueError(
                    f"Row {tx.row}: oversell of {tx.ticker}; held {pos.shares}, selling {tx.quantity}."
                )
            removed = (
                pos.basis if tx.quantity == pos.shares else pos.basis * tx.quantity / pos.shares
            )
            proceeds = tx.quantity * tx.price - tx.fees
            pos.shares -= tx.quantity
            pos.basis -= removed
            pos.realized += proceeds - removed
            cashflows.append((tx.date, proceeds))
    return positions, cashflows, notices
