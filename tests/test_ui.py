from pathlib import Path

from streamlit.testing.v1 import AppTest
from test_core import snapshot, tx

from portfolio.analytics import analyze


def test_app_boots_without_network_or_key():
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run(timeout=20)
    assert not app.exception
    assert app.title[0].value == "Portfolio Analyst"
    assert len(app.tabs) == 5
    next(button for button in app.button if button.label == "Try with sample data").click().run()
    assert not app.exception
    assert app.dataframe


def test_complete_dashboard_chat_and_clear():
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"))
    app.session_state["result"] = analyze([tx()], snapshot())
    app.session_state["benchmark_used"] = "SPY"
    app.run(timeout=20)
    assert not app.exception
    assert any(metric.label == "Market value" for metric in app.metric)
    app.text_input(key="groq_api_key").set_value("fake-session-key").run()
    next(button for button in app.button if button.label == "Explain my XIRR").click().run()
    assert not app.exception
    assert len(app.session_state["chat"]) == 2
    xirr_answer = app.session_state["chat"][-1]["content"]
    next(button for button in app.button if button.label == "How concentrated am I?").click().run()
    assert not app.exception
    assert len(app.session_state["chat"]) == 4
    assert app.session_state["chat"][-1]["content"] != xirr_answer
    assert "Effective holdings" in app.session_state["chat"][-1]["content"]
    assert app.text_input(key="groq_api_key").value == "fake-session-key"
    next(
        button for button in app.button if button.label == "Clear portfolio and chat"
    ).click().run()
    assert not app.exception
    assert "result" not in app.session_state
    assert app.text_input(key="groq_api_key").value == "fake-session-key"
