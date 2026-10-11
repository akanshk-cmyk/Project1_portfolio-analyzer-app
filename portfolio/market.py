"""Only public market data is cached; never cache a user's portfolio."""

import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol

import pandas as pd
import streamlit as st
import yfinance as yf


@dataclass
class MarketSnapshot:
    history: dict
    metadata: dict
    errors: dict
    fetched_at: str


class MarketDataProvider(Protocol):
    def fetch(self, tickers: tuple[str, ...], start: str) -> MarketSnapshot: ...


class YahooProvider:
    def fetch(self, tickers, start):
        errors, metadata, histories = {}, {}, {}
        batch = pd.DataFrame()
        for attempt in range(3):
            try:
                batch = yf.download(
                    list(tickers),
                    start=start,
                    auto_adjust=False,
                    actions=True,
                    progress=False,
                    group_by="ticker",
                    # yfinance initializes a shared SQLite timezone/cookie cache.
                    # Serial downloads avoid first-run database-lock races on Windows.
                    threads=False,
                    timeout=8,
                )
                if not batch.empty:
                    break
            except Exception:
                pass
            if attempt < 2:
                time.sleep(0.5 * 2**attempt)
        for symbol in tickers:
            try:
                info = yf.Ticker(symbol).get_info()
                currency = info.get("currency")
                kind = info.get("quoteType")
                metadata[symbol] = {
                    "currency": currency,
                    "sector": info.get("sector", "ETF / Unclassified"),
                    "quote_type": kind,
                    "symbol": info.get("symbol", symbol),
                }
                if currency != "USD":
                    errors[symbol] = (
                        f"USD currency could not be verified (reported {currency or 'unknown'})."
                    )
                elif kind not in {"EQUITY", "ETF"}:
                    errors[symbol] = "Only US equities and ETFs are supported."
                elif info.get("symbol", symbol) != symbol:
                    errors[symbol] = (
                        "Symbol changed; reconcile corporate actions before proceeding."
                    )
            except Exception:
                errors[symbol] = (
                    "Could not verify ticker/currency. Retry when Yahoo data is available."
                )
            try:
                frame = (
                    batch[symbol].copy()
                    if isinstance(batch.columns, pd.MultiIndex)
                    else batch.copy()
                )
                if "Close" not in frame or frame.Close.dropna().empty:
                    frame = pd.DataFrame()
            except KeyError:
                frame = pd.DataFrame()
            # A batch can succeed for only some tickers. Retry missing histories separately.
            if frame.empty:
                for attempt in range(2):
                    try:
                        frame = yf.Ticker(symbol).history(
                            start=start,
                            auto_adjust=False,
                            actions=True,
                            timeout=8,
                            raise_errors=True,
                        )
                        if not frame.empty:
                            break
                    except Exception:
                        frame = pd.DataFrame()
                    if attempt == 0:
                        time.sleep(0.5)
            try:
                if frame.empty:
                    histories[symbol] = frame
                    continue
                frame.index = pd.to_datetime(frame.index).tz_localize(None).normalize()
                frame = frame.dropna(subset=["Close"])
                if "Stock Splits" not in frame:
                    frame["Stock Splits"] = 0.0
                histories[symbol] = frame
            except (KeyError, ValueError):
                histories[symbol] = pd.DataFrame()
        return MarketSnapshot(histories, metadata, errors, datetime.now().astimezone().isoformat())


@st.cache_data(ttl=900, max_entries=64, show_spinner=False)
def fetch_market(tickers: tuple[str, ...], start: str):
    # Cache keys contain only ticker symbols and public-history start date.
    return YahooProvider().fetch(tickers, start)


def split_events(snapshot):
    return {
        ticker: [
            (day.date(), float(row["Stock Splits"]))
            for day, row in frame.iterrows()
            if row["Stock Splits"] > 0
        ]
        for ticker, frame in snapshot.history.items()
        if not frame.empty
    }


def market_checks(transactions, snapshot):
    errors, warnings = [], []
    splits = split_events(snapshot)
    for ticker in sorted({t.ticker for t in transactions}):
        if ticker in snapshot.errors:
            errors.append(f"{ticker}: {snapshot.errors[ticker]}")
        frame = snapshot.history.get(ticker, pd.DataFrame())
        if frame.empty:
            errors.append(
                f"{ticker}: no history available; unknown/delisted ticker cannot be verified."
            )
            continue
        # Large unexplained gaps may represent unsupported corporate actions. Require correction.
        gaps = frame.Close.pct_change().abs() > 0.60
        if gaps.any():
            errors.append(
                f"{ticker}: unexplained price gap above 60%; verify merger/spinoff history before analysis."
            )
        for tx in [t for t in transactions if t.ticker == ticker]:
            eligible = frame.loc[frame.index >= pd.Timestamp(tx.date)]
            if eligible.empty:
                errors.append(f"Row {tx.row}: no market close on or after trade date.")
                continue
            if pd.Timestamp(tx.date) < frame.index[0] - timedelta(days=7):
                errors.append(f"Row {tx.row}: trade precedes available listing/history.")
                continue
            point = eligible.iloc[0]
            factor = 1.0
            for day, ratio in splits.get(ticker, []):
                if day > tx.date:
                    factor *= ratio
            trade_price = float(tx.price) / factor
            if trade_price < point.Low * 0.999 or trade_price > point.High * 1.001:
                warnings.append(
                    f"Row {tx.row}: trade price outside the next available day's low/high range; check price and split convention."
                )
    return errors, warnings
