"""Bounded task dispatch shared by the Python and CLI evaluation entry points."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from itertools import islice
from typing import TypeVar

from inspect_robots.errors import ConfigError

_T = TypeVar("_T")
_R = TypeVar("_R")


# --- Argument validation helpers ---


def validate_max_workers(max_workers: int) -> None:
    """Reject invalid limits before constructing task resources."""
    if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers < 1:
        raise ConfigError("max_workers must be a positive integer")


# --- End argument validation helpers ---


def run_parallel(tasks: Sequence[_T], run: Callable[[_T], _R], max_workers: int) -> list[_R]:
    """Keep at most max_workers tasks in flight and return results in input order.

    A raised exception stops admission. Active tasks finish and release their
    resources before the exception propagates, including on KeyboardInterrupt.
    Threads cannot forcibly interrupt an active backend call.
    """
    results: dict[int, _R] = {}
    remaining = iter(enumerate(tasks))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        pending = {
            executor.submit(run, task): index for index, task in islice(remaining, max_workers)
        }
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            # Observe every completed exception before admitting another task.
            for future in done:
                results[pending.pop(future)] = future.result()
            for _ in done:
                item = next(remaining, None)
                if item is not None:
                    index, task = item
                    pending[executor.submit(run, task)] = index
    return [results[index] for index in range(len(tasks))]
