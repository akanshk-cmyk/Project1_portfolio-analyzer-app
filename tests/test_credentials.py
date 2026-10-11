from unittest.mock import patch

from portfolio.ai import environment_key


def test_key_comes_from_explicit_project_env_and_is_trimmed():
    with patch("portfolio.ai.dotenv_values", return_value={"GROQ_API_KEY": " fake-key "}) as read:
        assert environment_key() == "fake-key"
        assert read.call_args.args[0].name == ".env"
        assert read.call_args.args[0].parent.name == "portfolio-analyzer"


def test_empty_env_falls_back_to_process_configuration():
    with (
        patch("portfolio.ai.dotenv_values", return_value={}),
        patch.dict("os.environ", {"GROQ_API_KEY": "fake-env-key"}),
    ):
        assert environment_key() == "fake-env-key"
