"""Regex scans of page markup that stay linear in the page's length.

Each helper finds what the plain regex finds: a pattern that must end in a closing string
is run only up to that string's last occurrence, and a tag pattern is tried once per tag.
"""
from __future__ import annotations

import re
from typing import Callable, Iterator, Match, Optional, Pattern, Union


def last_end(text: str, closer: Union[str, Pattern]) -> int:
    """Just past the last `closer` in text, 0 when there is none."""
    if isinstance(closer, str):
        i = text.rfind(closer)
        return i + len(closer) if i >= 0 else 0
    end = 0
    for m in closer.finditer(text):
        end = m.end()
    return end


def sub_upto(pattern: Pattern, repl: Union[str, Callable], text: str,
             closer: Union[str, Pattern]) -> str:
    """pattern.sub for a pattern whose matches end in `closer`."""
    cut = last_end(text, closer)
    return pattern.sub(repl, text[:cut]) + text[cut:]


def search_upto(pattern: Pattern, text: str, closer: Union[str, Pattern]) -> Optional[Match]:
    return pattern.search(text, 0, last_end(text, closer))


def tag_matches(text: str, opener: Pattern, pattern: Pattern,
                endpos: Optional[int] = None) -> Iterator[Match]:
    """pattern.finditer(text) for a pattern that starts with a tag `opener` (`<a\\s[^>]*`,
    with no '>') finds, whose part inside the tag holds no '>'. A start inside a tag that
    failed fails too, so each tag is tried once."""
    endpos = len(text) if endpos is None else endpos
    pos = 0
    for tag in opener.finditer(text, 0, endpos):
        if tag.start() < pos:
            continue
        m = pattern.match(text, tag.start(), endpos)
        if m:
            yield m
            pos = max(m.end(), tag.start() + 1)


def tags(text: str, name: str) -> Iterator[str]:
    """Each tag `<name` starts, up to its '>' (`<name[^>]*>`); name is a regex."""
    pattern = re.compile(name + r"[^>]*>", re.I)
    for m in pattern.finditer(text, 0, last_end(text, ">")):
        yield m.group(0)
