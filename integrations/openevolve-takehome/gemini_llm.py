"""Direct Gemini worker adapter with cost accounting and trace hooks."""

from __future__ import annotations

import asyncio

import fcntl

import json

import logging

import os

import time

import uuid

from typing import Any

from openevolve.llm.base import LLMInterface

logger = logging.getLogger(__name__)

GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

DEFAULT_GEMINI_MODEL = "gemini-3.1-pro-preview"

GEMINI_API_KEY_ENV = "GEMINI_API_KEY"

GEMINI_PRICE = {
    "input_tokens": 2.00 / 1e6,
    "output_tokens": 12.00 / 1e6,
}

GEMINI_COST_BASIS = "gemini_list_price_estimate"

GEMINI_TRANSIENT_STATUSES = frozenset({429, 502, 503, 504})

GEMINI_TRANSIENT_RETRY_DELAY_SECONDS = 5.0


def _transient_http_status(exc: BaseException) -> int | None:
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    try:
        status = int(status)
    except (TypeError, ValueError):
        return None
    return status if status in GEMINI_TRANSIENT_STATUSES else None


class BudgetExhausted(RuntimeError):
    """Raised before any API call once the ledger reaches the budget."""


class GeminiCostLedger:
    """Cross-process cost accounting at Gemini list prices.

    A flock-guarded JSON file (``spent_usd``/``calls``/``usage``) shared by all
    workers; the runner's budget watchdog reads ``spent_usd``. Provider
    subclasses override the price table. ``cost_basis`` labels the estimate."""

    PRICE = GEMINI_PRICE
    USAGE_KEYS = tuple(GEMINI_PRICE)
    COST_BASIS = GEMINI_COST_BASIS

    def __init__(self, path: str | None = None, model: str | None = None):
        self.path = path or os.environ.get("OE_COST_LEDGER", "cost_ledger.json")

    def _read(self, f):
        f.seek(0)
        raw = f.read()
        if not raw.strip():
            return {
                "spent_usd": 0.0,
                "calls": 0,
                "usage": {k: 0 for k in self.USAGE_KEYS},
                "cost_basis": self.COST_BASIS,
            }
        return json.loads(raw)

    def add(self, usage: dict) -> float:
        """Record one call's usage at list price; returns new total spend."""
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, "a+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                data = self._read(f)
                for key in self.USAGE_KEYS:
                    n = int(usage.get(key) or 0)
                    data["usage"][key] = data["usage"].get(key, 0) + n
                call_cost = self.price(usage)
                data["spent_usd"] += call_cost
                data["calls"] += 1
                data["cost_basis"] = self.COST_BASIS
                data["updated_at"] = time.time()
                f.seek(0)
                f.truncate()
                json.dump(data, f, indent=2)
                return data["spent_usd"]
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def total(self) -> float:
        try:
            with open(self.path) as f:
                fcntl.flock(f, fcntl.LOCK_SH)
                try:
                    return self._read(f)["spent_usd"]
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
        except FileNotFoundError:
            return 0.0

    def budget(self) -> float:
        return float(os.environ.get("OE_BUDGET_USD", "400"))

    def price(self, usage: dict) -> float:
        return sum(int(usage.get(key) or 0) * rate for key, rate in self.PRICE.items())


class GeminiLLM(LLMInterface):
    """OpenEvolve LLM backend over the Gemini OpenAI-compatibility endpoint.

    Trajectory contract: ``llm_call_started`` (prompt/messages), the budget guard,
    ``llm_call_completed`` with usage + estimated list-price cost, and
    ``llm_call_failed``.
    SDK-internal retries are DISABLED (``max_retries=0``) so every retry is a
    visible ``generate_with_context`` attempt with its own trace records.
    Sampling parameters are omitted. Reasoning defaults are preserved unless
    a provider subclass exposes an explicit effort environment variable."""

    PROVIDER_NAME = "gemini"
    DISPLAY_NAME = "Gemini"
    ENDPOINT = GEMINI_OPENAI_BASE_URL
    API_KEY_ENV = GEMINI_API_KEY_ENV
    MODEL_ENV = "OE_GEMINI_MODEL"
    REASONING_EFFORT_ENV = "OE_GEMINI_REASONING_EFFORT"
    DEFAULT_MODEL = DEFAULT_GEMINI_MODEL
    TOKEN_LIMIT_PARAM = "max_tokens"
    LEDGER_CLASS = GeminiCostLedger
    COST_BASIS = GEMINI_COST_BASIS

    @staticmethod
    def _usage_from_response(raw_usage: Any) -> dict[str, int]:
        return {
            "input_tokens": int(getattr(raw_usage, "prompt_tokens", 0) or 0),
            "output_tokens": int(getattr(raw_usage, "completion_tokens", 0) or 0),
        }

    def __init__(self, model_cfg=None):
        from oe_trajectory import install_log_trace

        install_log_trace()
        self.model = (
            getattr(model_cfg, "name", None)
            or os.environ.get(self.MODEL_ENV)
            or self.DEFAULT_MODEL
        )
        self.system_message = getattr(model_cfg, "system_message", None)
        self.reasoning_effort = (
            os.environ.get(self.REASONING_EFFORT_ENV)
            if self.REASONING_EFFORT_ENV
            else None
        )
        self.max_tokens = getattr(model_cfg, "max_tokens", None) or 32000
        self.timeout = getattr(model_cfg, "timeout", None) or 2400
        self.retries = getattr(model_cfg, "retries", None)
        self.retries = 4 if self.retries is None else self.retries
        self.retry_delay = getattr(model_cfg, "retry_delay", None) or 5
        self.weight = getattr(model_cfg, "weight", 1.0)
        api_key = getattr(model_cfg, "api_key", None)
        resolved_key = api_key or os.environ.get(self.API_KEY_ENV)
        if not resolved_key:
            raise RuntimeError(
                f"{self.API_KEY_ENV} is required for {self.PROVIDER_NAME}"
            )
        client_factory = getattr(model_cfg, "client_factory", None)
        if client_factory is None:
            import openai

            client_factory = openai.OpenAI
        self.client = client_factory(
            base_url=self.ENDPOINT,
            api_key=resolved_key,
            timeout=float(self.timeout),
            max_retries=0,
        )
        self.ledger = self.LEDGER_CLASS(model=self.model)
        logger.info(
            f"{self.DISPLAY_NAME}LLM ready: model={self.model} max_tokens={self.max_tokens} "
            f"budget=${self.ledger.budget():.2f} (list-price estimates)"
        )

    async def generate(self, prompt: str, **kwargs) -> str:
        return await self.generate_with_context(
            system_message=self.system_message,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )

    async def generate_with_context(self, system_message, messages, **kwargs) -> str:
        loop = asyncio.get_event_loop()
        retries = kwargs.get("retries", self.retries)
        retry_delay = kwargs.get("retry_delay", self.retry_delay)
        failed_nontransient = 0
        while True:
            try:
                return await loop.run_in_executor(
                    None, lambda: self._call(system_message, messages)
                )
            except BudgetExhausted:
                raise  # never retry a budget stop
            except Exception as e:
                status = _transient_http_status(e)
                if status is not None:
                    logger.warning(
                        f"{self.DISPLAY_NAME} call returned HTTP {status}; "
                        f"retrying in {GEMINI_TRANSIENT_RETRY_DELAY_SECONDS:.1f}s: {e}"
                    )
                    await asyncio.sleep(GEMINI_TRANSIENT_RETRY_DELAY_SECONDS)
                    continue
                if failed_nontransient >= retries:
                    logger.error(f"All {retries + 1} attempts failed: {e}")
                    raise
                failed_nontransient += 1
                logger.warning(
                    f"{self.DISPLAY_NAME} call failed "
                    f"(attempt {failed_nontransient}/{retries + 1}): {e}"
                )
                await asyncio.sleep(retry_delay)

    def _call(self, system_message, messages) -> str:
        from oe_trajectory import (
            current_attempt,
            emit_event,
            prompt_sha256,
        )

        call_id = str(uuid.uuid4())
        call_started = time.time()
        normalized_messages = [
            {"role": m["role"], "content": m["content"]} for m in messages
        ]
        attempt = current_attempt()
        spent = self.ledger.total()
        budget = self.ledger.budget()
        prompt_hash = prompt_sha256(system_message, normalized_messages)
        trace_context = {
            **attempt,
            "call_id": call_id,
            "provider": self.PROVIDER_NAME,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "prompt_sha256": prompt_hash,
        }
        emit_event(
            "llm_call_started",
            **trace_context,
            spent_before_usd=spent,
            budget_usd=budget,
            system_message=system_message or "",
            messages=normalized_messages,
        )
        if spent >= budget:
            emit_event(
                "llm_call_budget_rejected",
                **trace_context,
                spent_usd=spent,
                budget_usd=budget,
            )
            raise BudgetExhausted(
                f"Budget exhausted: ${spent:.2f} spent of ${budget:.2f} — refusing API call"
            )
        request_messages = (
            [{"role": "system", "content": system_message}] if system_message else []
        ) + normalized_messages
        # Deliberately no temperature/top_p/top_k. Reasoning is provider-default
        # unless a provider-specific explicit effort was configured.
        try:
            request = {"model": self.model, "messages": request_messages}
            request[self.TOKEN_LIMIT_PARAM] = self.max_tokens
            if self.reasoning_effort is not None:
                request["reasoning_effort"] = self.reasoning_effort
            response = self.client.chat.completions.create(**request)
        except BaseException as exc:
            emit_event(
                "llm_call_failed",
                **trace_context,
                duration_seconds=time.time() - call_started,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise

        raw_usage = getattr(response, "usage", None)
        usage = self._usage_from_response(raw_usage)
        total = self.ledger.add(usage)
        call_cost = self.ledger.price(usage)
        choice = response.choices[0]
        dump = getattr(response, "model_dump", None)
        response_dump = dump(mode="json") if callable(dump) else str(response)
        emit_event(
            "llm_call_completed",
            **trace_context,
            duration_seconds=time.time() - call_started,
            usage=usage,
            call_cost_usd=call_cost,
            cost_basis=self.COST_BASIS,
            total_spent_usd=total,
            stop_reason=getattr(choice, "finish_reason", None),
            response=response_dump,
        )
        logger.info(
            f"{self.DISPLAY_NAME} call: in={usage['input_tokens']} out={usage['output_tokens']} "
            f"finish={getattr(choice, 'finish_reason', None)} | "
            f"total spend ${total:.2f} / ${budget:.2f} (list-price estimate)"
        )
        text = choice.message.content
        if not text:
            raise RuntimeError(
                f"Empty text response (finish_reason={getattr(choice, 'finish_reason', None)})"
            )
        return text


def init_gemini_client(model_cfg):
    """Factory for LLMModelConfig.init_client (must stay module-level for pickling)."""
    return GeminiLLM(model_cfg)
