"""Direct DeepSeek worker adapter for OpenEvolve vanilla runs.

It reuses the existing traced OpenAI-compatible worker implementation while
changing only endpoint, credential, model, and provider-true price accounting.
Reasoning defaults are preserved unless an explicit runner effort is set;
sampling controls remain omitted.
"""

from __future__ import annotations

from typing import Any

from gemini_llm import GeminiCostLedger, GeminiLLM

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_API_KEY_ENV = "DEEPSEEK_API_KEY"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"
DEEPSEEK_COST_BASIS = "deepseek_list_price_estimate"
DEEPSEEK_MODEL_PRICES = {
    "deepseek-v4-flash": (0.14, 0.28),
    "deepseek-v4-pro": (0.435, 0.87),
}


def _model_prices(model: str) -> tuple[float, float]:
    try:
        return DEEPSEEK_MODEL_PRICES[model]
    except KeyError as exc:
        raise ValueError(f"no frozen DeepSeek price table for {model!r}") from exc


class DeepSeekCostLedger(GeminiCostLedger):
    USAGE_KEYS = ("input_tokens", "output_tokens")
    COST_BASIS = DEEPSEEK_COST_BASIS

    def __init__(
        self, path: str | None = None, model: str = DEFAULT_DEEPSEEK_MODEL
    ) -> None:
        super().__init__(path=path, model=model)
        self.model = model
        _model_prices(model)

    def price(self, usage: dict) -> float:
        input_rate, output_rate = _model_prices(self.model)
        return (
            int(usage.get("input_tokens") or 0) * input_rate
            + int(usage.get("output_tokens") or 0) * output_rate
        ) / 1e6


class DeepSeekLLM(GeminiLLM):
    PROVIDER_NAME = "deepseek"
    DISPLAY_NAME = "DeepSeek"
    ENDPOINT = DEEPSEEK_BASE_URL
    API_KEY_ENV = DEEPSEEK_API_KEY_ENV
    MODEL_ENV = "OE_DEEPSEEK_MODEL"
    REASONING_EFFORT_ENV = "OE_DEEPSEEK_REASONING_EFFORT"
    DEFAULT_MODEL = DEFAULT_DEEPSEEK_MODEL
    TOKEN_LIMIT_PARAM = "max_tokens"
    LEDGER_CLASS = DeepSeekCostLedger
    COST_BASIS = DEEPSEEK_COST_BASIS

    @staticmethod
    def _usage_from_response(raw_usage: Any) -> dict[str, int]:
        return {
            "input_tokens": int(getattr(raw_usage, "prompt_tokens", 0) or 0),
            "output_tokens": int(getattr(raw_usage, "completion_tokens", 0) or 0),
        }


def init_deepseek_client(model_cfg):
    return DeepSeekLLM(model_cfg)
