from unittest.mock import MagicMock, patch

from test_core import snapshot

from portfolio.market import YahooProvider


def test_partial_download_retries_missing_history():
    frame = snapshot().history["AAPL"]
    batch = frame.copy()
    batch.columns = __import__("pandas").MultiIndex.from_product([["SPY"], batch.columns])
    ticker = MagicMock()
    ticker.get_info.return_value = {"currency": "USD", "quoteType": "EQUITY"}
    ticker.history.return_value = frame
    with (
        patch("portfolio.market.yf.download", return_value=batch),
        patch("portfolio.market.yf.Ticker", return_value=ticker),
    ):
        result = YahooProvider().fetch(("AAPL", "SPY"), "2024-01-01")
    assert not result.errors
    assert not result.history["AAPL"].empty
    assert ticker.history.call_count == 1
