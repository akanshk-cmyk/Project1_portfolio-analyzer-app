from types import SimpleNamespace
from unittest.mock import patch

from portfolio.ai import GROQ_BASE_URL, available_models


def test_model_discovery_uses_official_endpoint():
    with patch("portfolio.ai.Groq") as client:
        client.return_value.models.list.return_value = SimpleNamespace(
            data=[SimpleNamespace(id="model-b"), SimpleNamespace(id="model-a")]
        )
        models, status = available_models(" dummy ")
        assert models == ["model-a", "model-b"]
        assert client.call_args.kwargs["base_url"] == GROQ_BASE_URL
        assert client.call_args.kwargs["api_key"] == "dummy"
        assert "loaded" in status


def test_model_discovery_without_key_does_not_call_provider():
    with patch("portfolio.ai.Groq") as client:
        assert available_models(" ")[0] == []
        client.assert_not_called()
