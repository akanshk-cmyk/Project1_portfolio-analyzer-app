"""Bounded CSV ingestion. Account identifiers and notes are discarded."""

import re
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from io import BytesIO

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

ALIASES = {
    "date": {"date", "trade_date", "transaction_date"},
    "ticker": {"ticker", "symbol", "stock"},
    "action": {"action", "side", "type", "transaction_type"},
    "quantity": {"quantity", "shares", "qty"},
    "price": {"price", "trade_price", "price_per_share"},
    "fees": {"fees", "fee", "commission"},
    "time": {"time", "trade_time"},
}
REQUIRED = ("date", "ticker", "action", "quantity", "price")


class Transaction(BaseModel):
    model_config = ConfigDict(frozen=True)
    date: date
    ticker: str
    action: str
    quantity: Decimal = Field(gt=0, max_digits=24, decimal_places=6)
    price: Decimal = Field(gt=0)
    fees: Decimal = Field(default=Decimal(0), ge=0)
    row: int
    time: str = "00:00:00"


def read_csv(data: bytes) -> pd.DataFrame:
    if len(data) > 5 * 1024 * 1024:
        raise ValueError("Upload exceeds the 5 MB limit.")
    if b"\x00" in data:
        raise ValueError("Upload must be a UTF-8 text CSV, not a binary file.")
    try:
        frame = pd.read_csv(
            BytesIO(data),
            dtype=str,
            keep_default_na=False,
            encoding="utf-8-sig",
            nrows=10001,
        )
    except (UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError("Cannot read CSV. Use a UTF-8 CSV with a header row.") from exc
    if len(frame) > 10000 or frame.empty:
        raise ValueError("CSV must contain between 1 and 10,000 transactions.")
    return frame


def detect_mapping(columns):
    mapping = {}
    for target, aliases in ALIASES.items():
        found = [c for c in columns if str(c).strip().lower().replace(" ", "_") in aliases]
        if len(found) == 1:
            mapping[target] = found[0]
    return mapping


def validate(frame, mapping, today=None):
    today = today or datetime.now().astimezone().date()
    errors, warnings, transactions = [], [], []
    missing = [c for c in REQUIRED if not mapping.get(c)]
    if missing:
        return [], ["Map required columns: " + ", ".join(missing)], []
    if len(set(mapping.values())) != len(mapping):
        return [], ["Each source column can be mapped only once."], []
    seen = set()
    for idx, record in enumerate(frame.to_dict("records"), 2):
        try:
            raw = {key: str(record[col]).strip() for key, col in mapping.items()}
            # ISO and US month/day/year formats only; avoid silently interpreting ambiguous dates.
            parsed = None
            for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y"):
                try:
                    parsed = datetime.strptime(raw["date"], fmt).date()
                    break
                except ValueError:
                    pass
            if parsed is None:
                raise ValueError("date must be YYYY-MM-DD or US MM/DD/YYYY")
            if parsed > today:
                raise ValueError("future trade date")
            ticker = raw["ticker"].upper()
            if not re.fullmatch(r"[A-Z][A-Z0-9.\-^]{0,14}", ticker):
                raise ValueError("invalid ticker format")
            action = raw["action"].upper()
            if action not in {"BUY", "SELL"}:
                raise ValueError(
                    "action must be BUY or SELL; transfers/corporate actions unsupported"
                )
            trade_time = raw.get("time", "") or "00:00:00"
            trade_time = datetime.strptime(trade_time, "%H:%M:%S").strftime("%H:%M:%S")
            values = {k: Decimal(raw.get(k, "0") or "0") for k in ("quantity", "price", "fees")}
            if not all(v.is_finite() for v in values.values()):
                raise ValueError("numeric values must be finite")
            tx = Transaction(
                date=parsed,
                ticker=ticker,
                action=action,
                row=idx,
                time=trade_time,
                **values,
            )
            if action == "SELL" and tx.fees > tx.quantity * tx.price:
                raise ValueError("sale fees exceed gross sale proceeds")
            signature = (
                tx.date,
                tx.time,
                ticker,
                action,
                tx.quantity,
                tx.price,
                tx.fees,
            )
            if signature in seen:
                warnings.append(
                    f"Row {idx}: possible duplicate; retained until you correct the CSV."
                )
            seen.add(signature)
            if parsed.year < 1990:
                warnings.append(f"Row {idx}: very old trade date; verify available market history.")
            transactions.append(tx)
        except (ValueError, InvalidOperation) as exc:
            errors.append(f"Row {idx}: {exc}")
    transactions.sort(key=lambda t: (t.date, t.time, t.row))
    return transactions, errors, warnings


def safe_csv(frame):
    copy = frame.copy()
    for col in copy.columns:
        copy[col] = copy[col].map(
            lambda v: (
                "'" + v if isinstance(v, str) and v.lstrip().startswith(("=", "+", "-", "@")) else v
            )
        )
    return copy.to_csv(index=False).encode("utf-8")
