# Personal Portfolio Analyst

Streamlit US equity/ETF analytics using yfinance and Groq. Decimal average cost, partial sales, splits, XIRR, same-flow benchmark, historical charts and AI-selected verified facts.

## Run with uv

```powershell
cd C:\code\GenAcademy\Week1_project\portfolio-analyzer
uv sync --locked
Copy-Item .env.example .env
# Set GROQ_API_KEY in .env, or enter it in the sidebar.
uv run streamlit run app.py
```

Open http://localhost:8501. Choose **Try with sample data**, or upload `../Sample Stock data.csv`. Both transaction_type and action headers work. Your existing uv demo is untouched. uv manages Python 3.12 and .venv; do not use pip or manually create environments. AI is opt-in; the app works without a key.

```powershell
uv run pytest -q
uv run ruff check .
uv run pip-audit
```

Only public market data is cached across sessions. Refresh after changing benchmark or concentration settings. Enter original trade-time shares for splits. Fees included, dividends/taxes excluded. Chart windows retain full-ledger return denominators. See [methodology](METHODOLOGY.md) and [open launch gates](GOVERNANCE.md).

This local build is not cleared for public production use. No Excel-certified golden set, broker reconciliations, live Groq credentials, licensed feed or external reviews are bundled. Corporate-action checks are heuristic.

References: [Groq SDK](https://github.com/groq/groq-python), [yfinance history API](https://ranaroussi.github.io/yfinance/reference/yfinance.price_history.html).
