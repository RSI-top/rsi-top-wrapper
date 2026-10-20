"""Lossless physical-call tracing for Shinka's native Gemini provider.

The vendored provider already builds valid Google GenAI requests, extracts
thought/content blocks, prices usage through Shinka's frozen catalog, and
retries provider failures.  This runtime patch preserves those semantics while
putting one ``llm_call_started`` plus one completed/failed event around every
physical request attempt.  Because the backoff decorator wraps the traced
function, every retry gets a new call id and its own complete request/response
record instead of being hidden inside one logical proposal.

No vendored Shinka source is modified.  The API key stays solely in the Google
client and never enters an emitted request or response.
"""

from __future__ import annotations

import logging
import time
import traceback
import uuid
from typing import Any

import backoff

logger = logging.getLogger(__name__)

GEMINI_RETRY_DELAY_SECONDS = 5.0
GEMINI_TRANSIENT_STATUSES = frozenset({429, 502, 503, 504})


def _transient_http_status(exc: BaseException) -> int | None:
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    try:
        status = int(status)
    except (TypeError, ValueError):
        return None
    return status if status in GEMINI_TRANSIENT_STATUSES else None


def _request(
    provider: Any,
    *,
    model: str,
    msg: str,
    system_msg: str,
    msg_history: list,
    kwargs: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], Any]:
    contents = provider.gemini_build_contents(msg_history, msg)
    temperature = kwargs.get("temperature", 0.8)
    top_p = kwargs.get("top_p", 1.0)
    max_tokens = kwargs.get("max_tokens", kwargs.get("max_output_tokens", 2048))
    thinking_budget = kwargs.get("thinking_budget", 0)
    generation_config = provider.types.GenerateContentConfig(
        temperature=float(temperature),
        top_p=float(top_p),
        max_output_tokens=int(max_tokens),
        system_instruction=system_msg if system_msg else None,
        automatic_function_calling=provider.build_gemini_afc_config(),
        thinking_config=provider.build_gemini_thinking_config(thinking_budget),
    )
    request = {
        "model": model,
        "contents": contents,
        "config": generation_config,
    }
    return request, contents, generation_config


def _result(
    provider: Any,
    *,
    response: Any,
    model: str,
    msg: str,
    system_msg: str,
    msg_history: list,
    kwargs: dict[str, Any],
    model_posteriors: dict[str, float] | None,
):
    from shinka.llm.providers.result import QueryResult

    thought, content = provider.gemini_extract_thoughts_and_content(response)
    new_msg_history = msg_history + [
        {"role": "user", "content": msg},
        {"role": "assistant", "content": content},
    ]
    return QueryResult(
        content=content,
        msg=msg,
        system_msg=system_msg,
        new_msg_history=new_msg_history,
        model_name=model,
        kwargs=kwargs,
        **provider.get_gemini_costs(response, model),
        thought=thought,
        model_posteriors=model_posteriors,
    )


def _started(model: str, request: dict[str, Any]):
    from trajectory import current_attempt, emit_event, prompt_sha256

    call_id = str(uuid.uuid4())
    context = current_attempt()
    emit_event(
        "llm_call_started",
        **context,
        call_id=call_id,
        provider="google-genai",
        model=model,
        request=request,
        prompt_sha256=prompt_sha256(
            request["config"].system_instruction,
            request["contents"],
        ),
    )
    return call_id, time.time(), context


def _completed(
    *,
    call_id: str,
    started: float,
    context: dict[str, Any],
    model: str,
    request: dict[str, Any],
    response: Any,
    result: Any,
) -> None:
    from trajectory import emit_event

    emit_event(
        "llm_call_completed",
        **context,
        call_id=call_id,
        provider="google-genai",
        model=model,
        duration_seconds=time.time() - started,
        request=request,
        response=response,
        result=result.to_dict(),
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        thinking_tokens=result.thinking_tokens,
        input_cost_usd=result.input_cost,
        output_cost_usd=result.output_cost,
        call_cost_usd=result.cost,
        cost_source="shinka_local_pricing_catalog",
    )


def _failed(
    *,
    call_id: str,
    started: float,
    context: dict[str, Any],
    model: str,
    request: dict[str, Any],
    exc: BaseException,
) -> None:
    from trajectory import emit_event

    emit_event(
        "llm_call_failed",
        **context,
        call_id=call_id,
        provider="google-genai",
        model=model,
        duration_seconds=time.time() - started,
        request=request,
        error_type=type(exc).__name__,
        error=str(exc),
        status_code=getattr(exc, "status_code", None) or getattr(exc, "code", None),
        traceback="".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        ),
    )


def _retry_event(details: dict[str, Any]) -> None:
    from trajectory import current_attempt, emit_event

    exc = details.get("exception")
    emit_event(
        "llm_call_retry_scheduled",
        **current_attempt(),
        provider="google-genai",
        failed_attempt=details.get("tries"),
        delay_seconds=details.get("wait"),
        error_type=type(exc).__name__ if exc is not None else None,
        error=str(exc) if exc is not None else None,
        status_code=(
            getattr(exc, "status_code", None) or getattr(exc, "code", None)
            if exc is not None
            else None
        ),
    )
    # Preserve the vendored provider's normal log message too.
    try:
        import shinka.llm.providers.gemini as provider

        provider.backoff_handler(details)
    except Exception:  # pragma: no cover - trace emission must not break retry
        logger.exception("failed to emit the stock Gemini retry log")


def _patch_queries() -> None:
    import shinka.llm.providers.gemini as provider
    import shinka.llm.query as query_module

    # Provider throttling and transient gateway failures must not turn into
    # failed search attempts.  The outer 12-hour run timeout remains the bound.
    retry_transient = backoff.on_exception(
        backoff.constant,
        (Exception,),
        interval=GEMINI_RETRY_DELAY_SECONDS,
        jitter=None,
        giveup=lambda exc: _transient_http_status(exc) is None,
        on_backoff=_retry_event,
    )
    retry_other = backoff.on_exception(
        backoff.expo,
        (Exception,),
        max_tries=provider.MAX_TRIES,
        max_value=provider.MAX_VALUE,
        max_time=provider.MAX_TIME,
        on_backoff=_retry_event,
    )

    @retry_other
    @retry_transient
    def query_gemini(
        client,
        model,
        msg,
        system_msg,
        msg_history,
        output_model,
        model_posteriors=None,
        **kwargs,
    ):
        if output_model is not None:
            raise ValueError("Gemini does not support structured output.")
        request, _, _ = _request(
            provider,
            model=model,
            msg=msg,
            system_msg=system_msg,
            msg_history=msg_history,
            kwargs=kwargs,
        )
        call_id, started, context = _started(model, request)
        try:
            response = client.models.generate_content(**request)
            result = _result(
                provider,
                response=response,
                model=model,
                msg=msg,
                system_msg=system_msg,
                msg_history=msg_history,
                kwargs=kwargs,
                model_posteriors=model_posteriors,
            )
        except BaseException as exc:
            _failed(
                call_id=call_id,
                started=started,
                context=context,
                model=model,
                request=request,
                exc=exc,
            )
            raise
        _completed(
            call_id=call_id,
            started=started,
            context=context,
            model=model,
            request=request,
            response=response,
            result=result,
        )
        return result

    @retry_other
    @retry_transient
    async def query_gemini_async(
        client,
        model,
        msg,
        system_msg,
        msg_history,
        output_model,
        model_posteriors=None,
        **kwargs,
    ):
        if output_model is not None:
            raise ValueError("Gemini does not support structured output.")
        request, _, _ = _request(
            provider,
            model=model,
            msg=msg,
            system_msg=system_msg,
            msg_history=msg_history,
            kwargs=kwargs,
        )
        call_id, started, context = _started(model, request)
        try:
            response = await client.aio.models.generate_content(**request)
            result = _result(
                provider,
                response=response,
                model=model,
                msg=msg,
                system_msg=system_msg,
                msg_history=msg_history,
                kwargs=kwargs,
                model_posteriors=model_posteriors,
            )
        except BaseException as exc:
            _failed(
                call_id=call_id,
                started=started,
                context=context,
                model=model,
                request=request,
                exc=exc,
            )
            raise
        _completed(
            call_id=call_id,
            started=started,
            context=context,
            model=model,
            request=request,
            response=response,
            result=result,
        )
        return result

    provider.query_gemini = query_gemini
    provider.query_gemini_async = query_gemini_async
    query_module.query_gemini = query_gemini
    query_module.query_gemini_async = query_gemini_async
    import shinka.llm.providers as providers_package

    providers_package.query_gemini = query_gemini
    providers_package.query_gemini_async = query_gemini_async


def apply() -> None:
    _patch_queries()
    logger.info("Gemini 3.1 Pro physical-call tracing patch applied")
