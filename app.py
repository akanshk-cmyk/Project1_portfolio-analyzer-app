"""Run using uv run streamlit run app.py."""

import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from portfolio.ai import (
    DEFAULT_MODEL,
    available_models,
    environment_key,
    explain,
    fallback,
    money,
    pct,
)
from portfolio.analytics import analyze
from portfolio.holdings import reconstruct
from portfolio.ingestion import REQUIRED, detect_mapping, read_csv, safe_csv, validate
from portfolio.market import fetch_market, market_checks, split_events

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
if "pending_groq_model" in st.session_state:
    st.session_state["groq_model"] = st.session_state.pop("pending_groq_model")
st.set_page_config(page_title="Portfolio Analyst", page_icon="📊", layout="wide")
st.title("Portfolio Analyst")
st.caption(
    "Personal US portfolio analytics · Informational only, not investment, tax, or legal advice"
)
st.caption(
    "Delayed market data · USD only · Price return excludes dividends · Trade fees included; taxes excluded"
)
with st.sidebar:
    st.header("Analyst settings")
    enabled = st.toggle("Enable Groq AI", value=False)
    if "groq_api_key" not in st.session_state:
        st.session_state["groq_api_key"] = environment_key()
    key = st.text_input("Groq API key", type="password", key="groq_api_key").strip()
    st.caption(
        "API key loaded"
        if key
        else "No API key loaded. Enter it here or set GROQ_API_KEY in the app's .env file."
    )
    model = st.text_input(
        "Groq model", value=os.getenv("GROQ_MODEL", DEFAULT_MODEL), key="groq_model"
    )
    if st.button("Load Groq models"):
        with st.spinner("Checking models available from Groq…"):
            models, model_status = available_models(key)
        st.session_state["groq_models"] = models
        st.session_state["groq_model_status"] = model_status
    if st.session_state.get("groq_model_status"):
        st.caption(st.session_state["groq_model_status"])
    if st.session_state.get("groq_models"):
        models = st.session_state["groq_models"]
        preferred = model if model in models else DEFAULT_MODEL
        choice = st.selectbox(
            "Available Groq model",
            models,
            index=models.index(preferred) if preferred in models else 0,
        )
        if st.button("Use selected model"):
            st.session_state["pending_groq_model"] = choice
            st.session_state.pop("summary", None)
            st.rerun()
    benchmark = st.selectbox("Same-flow benchmark", ["SPY", "QQQ", "DIA", "IWM"])
    threshold = st.slider("Single-holding flag (%)", 5, 50, 10) / 100
    limit = st.number_input("Effective holdings threshold", 1, 50, 10)
    st.caption(
        "Only aggregated holdings and metrics plus your question go to Groq. Account/notes columns are discarded. Verify provider retention/no-training terms before enabling AI."
    )
    st.caption("Session-only portfolio. Public market cache: 15 minutes.")
    if st.button("Clear portfolio and chat"):
        for state_key in list(st.session_state):
            if not state_key.startswith(("groq_", "pending_groq_")):
                del st.session_state[state_key]
        st.rerun()


def clear_analysis():
    for item in ("result", "summary", "chat", "ai_steps", "ai_tokens"):
        st.session_state.pop(item, None)


def compute(txs, refresh=False, overrides=None):
    if refresh:
        fetch_market.clear()
    with st.spinner("Fetching and validating delayed market data…"):
        snap = fetch_market(tuple(sorted({t.ticker for t in txs} | {benchmark})), str(txs[0].date))
        errors, warnings = market_checks(txs, snap)
        for warning in warnings[:50]:
            st.warning(warning)
        try:
            reconstruct(txs, split_events(snap))
        except ValueError as exc:
            errors.append(str(exc))
        if errors:
            for error in errors[:50]:
                st.error(error)
            return
        try:
            result = analyze(txs, snap, benchmark, overrides, threshold, limit)
        except ValueError as exc:
            st.error(str(exc))
            return
        st.session_state.update(
            result=result, transactions=txs, snapshot=snap, benchmark_used=benchmark
        )
        st.session_state.pop("summary", None)
        st.session_state.pop("chat", None)


def ask(question, chat=False):
    result = st.session_state["result"]
    steps, tokens = (
        st.session_state.get("ai_steps", 0),
        st.session_state.get("ai_tokens", 0),
    )
    if not enabled or steps >= 20 or tokens >= 20000:
        reason = (
            "Groq AI is disabled. Enable it in the sidebar for AI-selected answers."
            if not enabled
            else "AI session budget reached. Clear the portfolio to start a new session."
        )
        return fallback(result, question if chat else None), {
            "status": reason + " Showing question-specific computed facts."
        }
    answer, trace = explain(result, question, key, model, chat)
    st.session_state["ai_steps"] = steps + 1
    st.session_state["ai_tokens"] = tokens + trace.get("tokens", 0)
    return answer, trace


upload, portfolio, performance, chat_tab, methodology = st.tabs(
    ["Upload", "Portfolio", "Performance", "AI Analyst", "Methodology"]
)
with upload:
    st.subheader("Import your transaction ledger")
    st.caption(
        "Required: date, ticker, action, quantity, price. Optional fees/time. Dates: YYYY-MM-DD or US MM/DD/YYYY. Use original trade-time shares and prices for split handling."
    )
    sample = (ROOT / "data/sample.csv").read_bytes()
    st.download_button("Download sample CSV", sample, "portfolio_template.csv", "text/csv")
    file = st.file_uploader("Transaction CSV · maximum 5 MB / 10,000 rows", type=["csv"])
    if st.button("Try with sample data"):
        st.session_state["sample"] = True
    data = file.getvalue() if file else sample if st.session_state.get("sample") else None
    if data:
        try:
            frame = read_csv(data)
            detected, mapping = detect_mapping(frame.columns), {}
            with st.expander("Column mapping", expanded=not all(c in detected for c in REQUIRED)):
                for target in (*REQUIRED, "fees", "time"):
                    choices = ["(unmapped)", *frame.columns]
                    selected = st.selectbox(
                        target,
                        choices,
                        index=choices.index(detected[target]) if target in detected else 0,
                        key=f"map_{target}_{','.join(frame.columns)}",
                    )
                    if selected != "(unmapped)":
                        mapping[target] = selected
            txs, errors, warnings = validate(frame, mapping)
            st.caption(
                "Whitespace trimmed; tickers/actions uppercased; account and notes discarded. Same-day ordering uses time (HH:MM:SS) if supplied, otherwise file order."
            )
            if txs:
                preview = pd.DataFrame(
                    [{k: str(v) for k, v in t.model_dump().items()} for t in txs]
                )
                date_times = []
                for trade in txs:
                    source_time = (
                        str(frame.iloc[trade.row - 2][mapping["time"]]).strip()
                        if "time" in mapping
                        else ""
                    )
                    date_times.append(
                        f"{trade.date} {trade.time}"
                        if source_time
                        else f"{trade.date} (time not supplied)"
                    )
                preview = preview.drop(columns=["date", "time"])
                preview.insert(1, "date and time", date_times)
                st.dataframe(preview.head(50), hide_index=True, width="stretch")
                st.caption(
                    f"{len(txs):,} rows · {txs[0].date} to {txs[-1].date} · {len({t.ticker for t in txs})} tickers"
                )
            for error in errors[:50]:
                st.error(error)
            for warning in warnings[:50]:
                st.warning(warning)
            st.caption(
                "Analyze runs ticker, USD currency, historical-range, split and oversell checks. Blocking issues require a corrected CSV."
            )
            if st.button("Analyze portfolio", type="primary", disabled=bool(errors) or not txs):
                clear_analysis()
                compute(txs)
                if "result" in st.session_state:
                    st.success("Portfolio validated. Open Portfolio or Performance to explore.")
        except ValueError as exc:
            st.error(str(exc))

result = st.session_state.get("result")
with portfolio:
    if result is None:
        st.info("Upload a ledger or try the sample to view your portfolio.")
    else:
        st.subheader("Consolidated portfolio")
        st.caption(
            f"As of market close {result['as_of']} · fetched {result['fetched_at']} · {'STALE' if result['stale'] else 'delayed data'}"
        )
        if st.session_state.get("benchmark_used") != benchmark:
            st.info("Refresh to apply the new benchmark.")
        if st.button("Refresh market data and settings"):
            compute(st.session_state["transactions"], refresh=True)
            st.rerun()
        for notice in result["notices"]:
            st.info(notice)
        if result["missing"]:
            st.error(
                "Missing prices: " + ", ".join(result["missing"]) + ". Gains and XIRR are N/A."
            )
            with st.form("manual"):
                overrides = {
                    t: st.number_input(
                        f"{t} user-entered last price USD", min_value=0.0, format="%.6f"
                    )
                    for t in result["missing"]
                }
                if st.form_submit_button("Apply user-entered valuations"):
                    st.session_state["result"] = analyze(
                        st.session_state["transactions"],
                        st.session_state["snapshot"],
                        benchmark,
                        overrides,
                        threshold,
                        limit,
                    )
                    st.session_state.pop("summary", None)
                    st.session_state.pop("chat", None)
                    st.rerun()
        m = result["metrics"]
        for col, label, field in zip(
            st.columns(5),
            [
                "Market value",
                "Cost basis",
                "Unrealized gain/loss",
                "Realized gain/loss",
                "Day price change",
            ],
            [
                "market_value",
                "cost_basis",
                "unrealized_gain",
                "realized_gain",
                "day_change",
            ],
        ):
            col.metric(label, money(m[field]))
        st.caption(
            "Realized gain/loss is not a tax figure. Day change uses current shares, excluding trade cash flows."
        )
        st.subheader("Portfolio explanation")
        if enabled and "summary" not in st.session_state:
            with st.spinner("Selecting verified portfolio facts…"):
                st.session_state["summary"] = ask(
                    "Summarize size, price return, concentration and strongest/weakest positions."
                )
        if st.button("Regenerate explanation"):
            st.session_state["summary"] = ask(
                "Summarize size, concentration and strongest/weakest positions with neutral facts."
            )
        text, trace = st.session_state.get(
            "summary", (fallback(result), {"status": "Deterministic key stats"})
        )
        st.write(text)
        st.caption(
            f"{trace['status']} · Sources: holdings, price-return metrics · Informational only, not investment advice"
        )
        holdings = pd.DataFrame(result["holdings"])
        opened = holdings[holdings.shares > 0]
        st.subheader("Allocation by holding, not look-through")
        allocation = st.radio("Allocation basis", ["Market value", "Cost basis"], horizontal=True)
        field = "market_value" if allocation == "Market value" else "cost_basis"
        positive = opened[opened[field].fillna(0) > 0]
        left, right = st.columns(2)
        if not positive.empty:
            left.plotly_chart(
                px.pie(positive, names="ticker", values=field, hole=0.65),
                width="stretch",
            )
            right.plotly_chart(
                px.bar(
                    positive.groupby("sector", as_index=False)[field].sum(),
                    x="sector",
                    y=field,
                ),
                width="stretch",
            )
        else:
            st.info("No positive-valued holdings to plot.")
        st.dataframe(
            opened[["ticker", field, "weight_pct", "sector"]],
            hide_index=True,
            width="stretch",
        )
        for flag in result["concentration"]["flags"]:
            st.warning(flag)
        st.subheader("Holdings")
        query = st.text_input("Filter tickers").upper().strip()
        filtered = opened[opened.ticker.str.contains(query, regex=False)]
        st.dataframe(filtered, hide_index=True, width="stretch")
        st.download_button("Export holdings CSV", safe_csv(filtered), "holdings.csv", "text/csv")
        with st.expander("Sold positions · realized gain/loss is not a tax figure"):
            st.dataframe(
                holdings[holdings.shares == 0][["ticker", "realized_gain"]],
                hide_index=True,
                width="stretch",
            )

with performance:
    if result is None:
        st.info("Analyze a ledger to view performance.")
    else:
        st.subheader("Historical price performance")
        st.caption(
            f"As of {result['as_of']} · Dividends excluded; trade fees included; taxes excluded · Sources: {', '.join(result['valuation_sources'])}"
        )
        m = result["metrics"]
        for col, label, field in zip(
            st.columns(4),
            [
                "Total invested",
                "Sale proceeds",
                "Total gain (price return)",
                "Cumulative price return",
            ],
            ["total_invested", "sale_proceeds", "total_gain", "cumulative_return_pct"],
        ):
            col.metric(label, pct(m[field]) if field.endswith("pct") else money(m[field]))
        period = st.radio("Chart range", ["1M", "6M", "YTD", "1Y", "All"], index=4, horizontal=True)
        history, end = result["history"], pd.Timestamp(result["as_of"])
        starts = {
            "1M": end - pd.DateOffset(months=1),
            "6M": end - pd.DateOffset(months=6),
            "YTD": pd.Timestamp(end.year, 1, 1),
            "1Y": end - pd.DateOffset(years=1),
            "All": history.index[0],
        }
        selected = history.loc[history.index >= starts[period]]
        st.caption(
            "Chart ranges filter values. Return metrics retain the full-ledger cash-flow basis and are not subperiod returns."
        )
        if result["history_days"] >= 365 and period in {"1Y", "All"}:
            st.metric(
                "Annualized XIRR (full-history price return)",
                pct(m["xirr_pct"]),
                help="Actual/365 rate equating dated buy/sale flows and terminal value.",
            )
        else:
            with st.expander("Annualized XIRR · interpret short histories cautiously"):
                st.warning(
                    "Annualizing short periods can mislead. Cumulative price return is the headline under one year."
                )
                st.write(pct(m["xirr_pct"]))
        st.caption(result["xirr_reason"])
        st.line_chart(selected[["portfolio_value", "net_invested"]])
        st.line_chart(selected[["cumulative_return_pct"]])
        with st.expander("Historical chart data (accessible table)"):
            st.dataframe(selected, width="stretch")
        contributions = pd.DataFrame(result["holdings"])
        contributions["gain_contribution"] = (
            contributions.unrealized_gain + contributions.realized_gain
        )
        st.plotly_chart(
            px.bar(
                contributions,
                x="ticker",
                y="gain_contribution",
                title="Contribution to total gain (USD)",
            ),
            width="stretch",
        )
        st.dataframe(
            contributions[["ticker", "gain_contribution"]],
            hide_index=True,
            width="stretch",
        )
        bench = result["benchmark"]
        st.subheader(f"{bench['ticker']} same-flow benchmark · price to price")
        if bench.get("eligible") and not bench.get("reason"):
            st.write(
                f"Cumulative price return: {pct(bench['cumulative_return_pct'])} · As of {bench['as_of']}"
            )
            if m["cumulative_return_pct"] is not None:
                st.write(
                    f"Portfolio minus benchmark: {m['cumulative_return_pct'] - bench['cumulative_return_pct']:.2f} percentage points."
                )
        else:
            st.info(
                bench.get("reason") or "Benchmark comparison requires at least 90 days of history."
            )
        with st.expander("Show XIRR calculation cash flows"):
            st.dataframe(pd.DataFrame(result["cashflows"]), hide_index=True, width="stretch")
            st.caption(
                "BUY negative; SELL net proceeds positive; open-position terminal value positive at last market close. Dividends = 0 in v1."
            )

with chat_tab:
    st.subheader("Ask your portfolio analyst")
    st.caption(
        "Read-only analytics tools. Verified facts rendered by Python. Raw CSV and account/notes fields are never sent to Groq; avoid personal identifiers in your question."
    )
    if result is None:
        st.info("Analyze a ledger first.")
    else:
        question = None
        for col, prompt in zip(
            st.columns(4),
            [
                "How concentrated am I?",
                "Which positions are dragging gains?",
                "Explain my XIRR",
                "Compare my portfolio with the benchmark",
            ],
        ):
            if col.button(prompt):
                question = prompt
        # Reserve the transcript above the input so newly rendered messages
        # remain above it on the submission rerun too.
        conversation = st.container()
        for message in st.session_state.get("chat", []):
            with conversation.chat_message(message["role"]):
                st.write(message["content"])
        question = (
            st.chat_input("Ask about holdings, concentration or price performance", max_chars=2000)
            or question
        )
        if question:
            with conversation.chat_message("user"):
                st.write(question)
            with conversation.chat_message("assistant"), st.spinner("Retrieving verified facts…"):
                answer, trace = ask(question, chat=True)
                st.write(answer)
                if trace["status"] != "verified":
                    st.info(trace["status"])
                st.caption(
                    f"{trace['status']} · Sources: holdings, concentration, cash-flow metrics · Informational only, not investment advice"
                )
                with st.expander("Execution metadata (no transaction contents)"):
                    st.json(trace)
            st.session_state.setdefault("chat", []).extend(
                [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ]
            )
        if st.session_state.get("chat"):
            conversation.feedback("thumbs", key=f"feedback_{len(st.session_state['chat'])}")
with methodology:
    st.subheader("Methodology and trust")
    st.markdown((ROOT / "METHODOLOGY.md").read_text(encoding="utf-8"))
