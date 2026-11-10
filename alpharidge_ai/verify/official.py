"""Official sites whose pages state the publish date only in the page body, so a date read
from the body counts as the publish date there and on no other site. The sites and how each
states its date are data: `data/official.json`.

A plain site entry takes a date written out in words after one of the shared date labels,
and on those sites a date guessed from the page also counts. A site entry written as an
object states its own date: `labels` (page text right before the date), `marks` (markup right
before the date) and `order` ("mdy" or "dmy", for numeric dates); a date is read from there
only, ISO dates included.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlsplit

_DATA = json.loads((Path(__file__).with_name("data") / "official.json").read_text(
    encoding="utf-8"))
DATE_LABELS = tuple(_DATA["date_labels"])


@dataclass(frozen=True)
class Rule:
    labels: Tuple[str, ...] = DATE_LABELS
    marks: Tuple[str, ...] = ()
    order: str = ""
    # A plain entry: a date guessed from the page counts too.
    guess: bool = True
    # Status-page incidents: the earliest Unix time in this attribute (seconds or
    # milliseconds), e.g. `data-datetime-unix`.
    unix: str = ""
    # The `datetime` attribute that follows this mark in the markup (e.g. a status page's
    # "Started" field): an exact time.
    time_mark: str = ""


def _rule(entry) -> Tuple[str, Rule]:
    if isinstance(entry, str):
        return entry, Rule()
    if set(entry) == {"site", "unix"} and entry["unix"]:
        attr = "data-datetime-unix" if entry["unix"] is True else str(entry["unix"])
        return entry["site"], Rule(unix=attr, guess=False)
    if set(entry) == {"site", "time_mark"} and entry["time_mark"]:
        return entry["site"], Rule(time_mark=entry["time_mark"], guess=False)
    order = entry.get("order", "")
    if order not in ("", "mdy", "dmy") or not (entry.get("labels") or entry.get("marks")):
        raise ValueError(f"official.json: bad entry {entry!r}")
    return entry["site"], Rule(tuple(entry.get("labels", ())), tuple(entry.get("marks", ())),
                               order, guess=False)


_RULES = [_rule(e) for e in _DATA["sites"]]
SITES = tuple(site for site, _ in _RULES)


def _split(site: str):
    host, _, path = site.partition("/")
    return host, "/" + path if path else ""


_PATHS: dict = {}
for _site, _r in _RULES:
    _host, _path = _split(_site)
    _PATHS.setdefault(_host, []).append((_path, _r))


def rule(url: str) -> Optional[Rule]:
    """How a page on one of the sites states its date, None elsewhere."""
    parts = urlsplit(url or "")
    host = (parts.hostname or "").lower().rstrip(".")
    host = host[4:] if host.startswith("www.") else host
    path = parts.path or "/"
    return next((r for p, r in _PATHS.get(host, ()) if path.startswith(p)), None)


def body_dated(url: str) -> bool:
    """A page on a plain site entry, where a date guessed from the page counts too."""
    r = rule(url)
    return r is not None and r.guess
