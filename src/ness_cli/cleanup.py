"""Attempt all CLI cleanup operations and preserve an existing failure."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from contextlib import asynccontextmanager

_logger = logging.getLogger(__name__)
Cleanup = Callable[[], Awaitable[None]]


async def cleanup_all(
    operations: Iterable[tuple[str, Cleanup]],
    *,
    primary_error: BaseException | None = None,
) -> None:
    """Run in order, reporting failures only after every operation is attempted.

    An existing error or cancellation remains primary. Otherwise, propagate a
    single cleanup error unchanged, or group multiple ordinary cleanup errors.
    """
    failures: list[tuple[str, BaseException]] = []
    for name, cleanup in operations:
        try:
            await cleanup()
        except BaseException as error:
            failures.append((name, error))

    if not failures:
        return

    # A cleanup cancellation must remain a cancellation, even if later cleanup
    # raises an ordinary error. KeyboardInterrupt/SystemExit follow the same rule.
    if primary_error is None:
        primary_error = next(
            (error for _, error in failures if not isinstance(error, Exception)),
            None,
        )
        propagate_primary = primary_error is not None
    else:
        propagate_primary = False

    if primary_error is not None:
        for name, error in failures:
            diagnostic = f"Failed to clean up {name}: {error}"
            primary_error.add_note(diagnostic)
            _logger.warning("%s", diagnostic)
        if propagate_primary:
            raise primary_error
        return

    for name, error in failures:
        error.add_note(f"During cleanup of {name}.")
    if len(failures) == 1:
        raise failures[0][1]

    details = "; ".join(f"{name}: {error}" for name, error in failures)
    raise BaseExceptionGroup(
        f"Cleanup failed: {details}", [error for _, error in failures]
    )


@asynccontextmanager
async def cleanup_on_exit(name: str, cleanup: Cleanup) -> AsyncIterator[None]:
    primary_error: BaseException | None = None
    try:
        yield
    except BaseException as error:
        primary_error = error
        raise
    finally:
        await cleanup_all(((name, cleanup),), primary_error=primary_error)
