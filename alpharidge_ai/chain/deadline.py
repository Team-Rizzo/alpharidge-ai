"""Chain calls with a deadline: a socket that stops answering costs one call, not the loop."""
from __future__ import annotations

import threading
from typing import Callable, TypeVar

T = TypeVar("T")


class ChainTimeout(TimeoutError):
    pass


def bounded(fn: Callable[[], T], seconds: float, what: str = "chain call") -> T:
    """`fn()` on a helper thread; ChainTimeout when it has not returned within `seconds`.
    The helper is left behind if it never returns: closing its connection ends it."""
    out: list = []
    done = threading.Event()

    def run() -> None:
        try:
            out.append((True, fn()))
        except BaseException as e:  # handed back to the caller below
            out.append((False, e))
        finally:
            done.set()

    threading.Thread(target=run, daemon=True, name=f"bounded:{what}").start()
    if not done.wait(seconds):
        raise ChainTimeout(f"{what} took over {seconds:.0f}s")
    ok, value = out[0]
    if not ok:
        raise value
    return value
