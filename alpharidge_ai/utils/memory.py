"""Process memory, split into anonymous memory and file-backed pages.

The entity-linking tables are memory-mapped, so resident size climbs as their pages are
read and is not a leak; anonymous memory is the number to watch.
"""
from __future__ import annotations

import os
import threading
from typing import Dict, Optional

_FIELDS = ("Rss", "Anonymous", "Pss_File")


def read(path: str = "/proc/self/smaps_rollup") -> Optional[Dict[str, int]]:
    """kB per field, or None where the kernel doesn't provide it."""
    try:
        with open(path) as f:
            out = {}
            for line in f:
                name, _, rest = line.partition(":")
                if name in _FIELDS:
                    out[name] = int(rest.split()[0])
            return out or None
    except (OSError, ValueError, IndexError):
        return None


def line() -> Optional[str]:
    m = read()
    if not m:
        return None
    gb = lambda k: m.get(k, 0) / 1048576
    return (f"[MEMORY] rss={gb('Rss'):.1f}GB anon={gb('Anonymous'):.1f}GB "
            f"file={gb('Pss_File'):.1f}GB threads={threading.active_count()} pid={os.getpid()}")
