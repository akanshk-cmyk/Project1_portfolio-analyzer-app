import json
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from groq import AuthenticationError
from test_core import snapshot, tx

from portfolio.ai import explain, fallback
from portfolio.analytics import analyze


def test_offline_answers_follow_question():
    result = analyze([tx()], snapshot())
    answers = [
        explain(result, q, "", "unused", chat=True)[0]
        for q in (
            "How concentrated am I?",
            "Explain my XIRR",
            "Compare my portfolio with the benchmark",
            "Which positions are dragging gains?",
        )
    ]
    assert len(set(answers)) == 4
    assert "Effective holdings" in answers[0]
    assert "XIRR uses dated" in answers[1]
    assert "same-flow benchmark" in answers[2]
    assert "AAPL:" in answers[3]
    assert "Portfolio market value" not in answers[0]


def test_unknown_offline_question_asks_for_clarification():
    result = analyze([tx()], snapshot())
    assert "Which of these" in fallback(result, "Tell me about something else")


def test_auth_failure_is_visible_without_exposing_provider_body():
    result = analyze([tx()], snapshot())
    response = httpx.Response(401, request=httpx.Request("POST", "https://api.groq.com"))
    error = AuthenticationError("secret-provider-body", response=response, body=None)
    with patch("portfolio.ai.Groq") as client:
        client.return_value.chat.completions.create.side_effect = error
        text, trace = explain(result, "Explain my XIRR", "dummy", "unused", chat=True)
    assert "XIRR uses dated" in text
    assert "HTTP 401" in trace["status"]
    assert "secret-provider-body" not in str(trace)


def test_final_model_request_keeps_question_and_focuses_facts():
    result = analyze([tx()], snapshot())
    replies = [
        SimpleNamespace(
            usage=None, choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=None))]
        ),
        SimpleNamespace(
            usage=None,
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps({"fact_ids": ["xirr"], "explanation_ids": ["xirr"]})
                    )
                )
            ],
        ),
    ]
    with patch("portfolio.ai.Groq") as client:
        create = client.return_value.chat.completions.create
        create.side_effect = replies
        text, trace = explain(result, "Explain my XIRR", "dummy", "unused", chat=True)
        final = json.loads(create.call_args.kwargs["messages"][-1]["content"])
    assert final["question"] == "Explain my XIRR"
    assert set(final["facts"]) == {"xirr"}
    assert trace["status"] == "verified"
    assert "XIRR uses dated" in text
