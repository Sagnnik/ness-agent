from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from random import uniform
from typing import Any

import httpx

from ness_cli.provider.codex.auth import CodexAuth

logger = logging.getLogger(__name__)

CODEX_RESPONSES_URL = "https://chatgpt.com/backend-api/codex/responses"
_MAX_BACKOFF_SECONDS = 8.0
_MAX_RETRY_DELAY_SECONDS = 60.0
_RETRYABLE_STREAM_ERROR_TYPES = {
    "rate_limit_error",
    "server_error",
    "service_unavailable_error",
}
_RETRYABLE_STREAM_ERROR_CODES = {
    "internal_server_error",
    "rate_limit_exceeded",
    "server_error",
    "server_is_overloaded",
}


class CodexStreamError(RuntimeError):
    """A terminal error delivered inside an otherwise successful SSE response."""

    def __init__(
        self,
        error: Any,
        *,
        retryable: bool,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(str(error))
        self.retryable = retryable
        self.retry_after = retry_after


async def _sleep(delay: float) -> None:
    await asyncio.sleep(delay)


def _json_hash(value: Any) -> str:
    rendered = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(rendered).hexdigest()


def _cache_snapshot(payload: dict[str, Any]) -> dict[str, Any] | None:
    cache_key = str(payload.get("prompt_cache_key") or "")
    if not cache_key:
        return None
    items = payload.get("input") or []
    return {
        "cache_key_hash": hashlib.sha256(cache_key.encode("utf-8")).hexdigest()[:12],
        "model": str(payload.get("model") or ""),
        "instructions_hash": _json_hash(payload.get("instructions") or ""),
        "tools_hash": _json_hash(payload.get("tools") or []),
        "input_item_hashes": tuple(_json_hash(item) for item in items),
    }


def _compare_cache_snapshots(
    current: dict[str, Any] | None,
    previous: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if current is None:
        return None
    current_items = current["input_item_hashes"]
    diagnostics: dict[str, Any] = {
        "cache_key_hash": current["cache_key_hash"],
        "current_input_items": len(current_items),
        "has_previous_request": previous is not None,
    }
    if previous is None:
        return diagnostics

    previous_items = previous["input_item_hashes"]
    matching_items = 0
    for old, new in zip(previous_items, current_items):
        if old != new:
            break
        matching_items += 1
    stable_configuration = (
        previous["model"] == current["model"]
        and previous["instructions_hash"] == current["instructions_hash"]
        and previous["tools_hash"] == current["tools_hash"]
    )
    diagnostics.update(
        {
            "prior_input_items": len(previous_items),
            "matching_input_items": matching_items,
            "stable_configuration": stable_configuration,
            "append_only_prefix": (
                stable_configuration and matching_items == len(previous_items)
            ),
        }
    )
    if matching_items < len(previous_items):
        diagnostics["first_mismatch_item"] = matching_items
    return diagnostics


def _error_detail(response: httpx.Response) -> str | None:
    try:
        payload = response.json()
    except (ValueError, json.JSONDecodeError):
        payload = None
    detail: Any = None
    if isinstance(payload, dict):
        detail = payload.get("detail")
        error = payload.get("error")
        if detail is None and isinstance(error, dict):
            detail = error.get("message") or error.get("detail")
        if detail is None and error is not None:
            detail = error
    if detail is None:
        detail = response.text.strip()
    if not detail:
        return None
    if not isinstance(detail, str):
        detail = json.dumps(detail, ensure_ascii=False, sort_keys=True)
    return detail[:2_000]


async def _raise_for_status(response: httpx.Response) -> None:
    if response.is_success:
        return
    await response.aread()
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        parts = [str(exc)]
        detail = _error_detail(response)
        if detail:
            parts.append(f"Backend detail: {detail}")
        request_id = response.headers.get("x-request-id") or response.headers.get(
            "x-oai-request-id"
        )
        if request_id:
            parts.append(f"Request ID: {request_id}")
        raise httpx.HTTPStatusError(
            "\n".join(parts),
            request=response.request,
            response=response,
        ) from exc


def _retry_after_seconds(headers: httpx.Headers) -> float | None:
    raw = headers.get("retry-after")
    if not raw:
        return None
    try:
        delay = float(raw)
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(raw)
        except (TypeError, ValueError, OverflowError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        delay = (retry_at - datetime.now(timezone.utc)).total_seconds()
    return delay if math.isfinite(delay) and delay >= 0 else None


def _retry_delay(retry_attempt: int, retry_after: float | None) -> float:
    base = (
        retry_after
        if retry_after is not None
        else min(2**retry_attempt, _MAX_BACKOFF_SECONDS)
    )
    jitter = uniform(0, min(1.0, max(base, 0.0) * 0.25))
    return min(base + jitter, _MAX_RETRY_DELAY_SECONDS)


def _stream_error(event: dict[str, Any], headers: httpx.Headers) -> CodexStreamError:
    error: Any = event.get("error")
    if error is None:
        response = event.get("response")
        if isinstance(response, dict):
            error = response.get("error")
    if error is None:
        error = event

    details = error if isinstance(error, dict) else event
    error_type = str(details.get("type") or "").casefold()
    error_code = str(details.get("code") or "").casefold()
    retryable = (
        error_type in _RETRYABLE_STREAM_ERROR_TYPES
        or error_code in _RETRYABLE_STREAM_ERROR_CODES
    )
    return CodexStreamError(
        error,
        retryable=retryable,
        retry_after=_retry_after_seconds(headers),
    )


def merge_streamed_response(
    completed: dict[str, Any] | None,
    output_items: list[dict[str, Any]],
    text_parts: list[str],
) -> dict[str, Any]:
    """Merge streamed content into the terminal response envelope.

    The ChatGPT Codex backend can emit a metadata/usage-only
    ``response.completed`` envelope. Returning that envelope verbatim drops
    the already-received text deltas, producing an empty AIMessage even though
    output-token usage is non-zero.
    """
    response = dict(completed or {})
    if output_items:
        # output_item.done contains complete replayable items (message,
        # reasoning, and function calls), so prefer it to a sparse envelope.
        response["output"] = [dict(item) for item in output_items]
    streamed_text = "".join(text_parts)
    if streamed_text:
        response["output_text"] = streamed_text
    return response


class CodexResponsesTransport:
    """Experimental, deliberately isolated ChatGPT Codex Responses transport."""

    def __init__(
        self, auth: CodexAuth, *, max_retries: int = 3, timeout: float = 180
    ) -> None:
        self.auth = auth
        self.max_retries = max_retries
        self.timeout = timeout
        self._cache_snapshots: dict[str, dict[str, Any]] = {}

    async def create(self, payload: dict[str, Any]) -> dict[str, Any]:
        snapshot = _cache_snapshot(payload)
        snapshot_key = snapshot["cache_key_hash"] if snapshot is not None else None
        previous_snapshot = (
            self._cache_snapshots.get(snapshot_key) if snapshot_key is not None else None
        )
        cache_diagnostics = _compare_cache_snapshots(snapshot, previous_snapshot)
        if cache_diagnostics is not None:
            logger.debug("Codex cache prefix diagnostics: %s", cache_diagnostics)
        refreshed_401 = False  # needs auth refresh
        retry_attempt = 0  # retry up to max_retries times
        while True:
            credentials = await self.auth.valid_credentials()
            headers = {
                "Authorization": f"Bearer {credentials.access_token}",
                "ChatGPT-Account-ID": credentials.account_id,
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
                "OpenAI-Beta": "responses=experimental",  # extra header for experimental features
                "originator": "ness-agent",
            }
            cache_session_id = str(payload.get("prompt_cache_key") or "")
            if cache_session_id:
                # Match the stable session-routing header used by the
                # first-party Codex client. Keep it identical to the body key
                # so the private backend sees one cache/session identity.
                headers["session_id"] = cache_session_id
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    # stream from the Codex Responses URL
                    async with client.stream(
                        "POST",
                        CODEX_RESPONSES_URL,
                        headers=headers,
                        json={**payload, "stream": True},
                    ) as response:
                        # handle 401 Unauthorized: likely due to token expiration
                        if response.status_code == 401 and not refreshed_401:
                            await response.aread()  # read the response body once
                            refreshed_401 = True  # can refresh only once
                            await self.auth.valid_credentials(force_refresh=True)
                            continue

                        await _raise_for_status(response)
                        completed: dict[str, Any] | None = None
                        output: list[dict[str, Any]] = []
                        text_parts: list[str] = []
                        async for line in response.aiter_lines():
                            if not line.startswith("data:"):  # ignore non-data lines
                                continue
                            raw = line[5:].strip()  # remove the "data:" prefix

                            if not raw or raw == "[DONE]":  # ignore empty or done lines
                                continue

                            event = json.loads(raw)  # parse the JSON event
                            kind = event.get("type")  # get the event type

                            # terminal response
                            if kind == "response.completed":
                                value = event.get("response")
                                if isinstance(value, dict):
                                    completed = value  # store the completed response

                            # one fully completed output item
                            elif kind == "response.output_item.done":
                                item = event.get("item")
                                if isinstance(item, dict):
                                    output.append(item)  # add to the output list

                            # streaming text delta
                            elif kind == "response.output_text.delta":
                                text_parts.append(
                                    str(event.get("delta") or "")
                                )  # gather the chunks

                            # final text chunk (fallback)
                            elif kind == "response.output_text.done" and not text_parts:
                                text_parts.append(str(event.get("text") or ""))

                            # error: HTTP is 200 but un-successful LLM response
                            elif kind in {"response.failed", "error"}:
                                raise _stream_error(event, response.headers)
                        merged = merge_streamed_response(
                            completed, output, text_parts
                        )  # combine all of them into one complete response
                        if snapshot_key is not None and snapshot is not None:
                            self._cache_snapshots[snapshot_key] = snapshot
                        if cache_diagnostics is not None:
                            merged["_cache_diagnostics"] = cache_diagnostics
                        return merged

            except (
                CodexStreamError,
                httpx.TransportError,
                httpx.HTTPStatusError,
                json.JSONDecodeError,
            ) as exc:
                retryable = isinstance(exc, httpx.TransportError)
                if isinstance(exc, httpx.HTTPStatusError):
                    retryable = (
                        exc.response.status_code == 429
                        or exc.response.status_code >= 500
                    )
                elif isinstance(exc, CodexStreamError):
                    retryable = exc.retryable
                if not retryable or retry_attempt >= self.max_retries:
                    raise
                retry_after = (
                    exc.retry_after
                    if isinstance(exc, CodexStreamError)
                    else (
                        _retry_after_seconds(exc.response.headers)
                        if isinstance(exc, httpx.HTTPStatusError)
                        else None
                    )
                )
                await _sleep(_retry_delay(retry_attempt, retry_after))
                retry_attempt += 1
