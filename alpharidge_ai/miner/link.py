"""Actor linking: names in the page text to entity IDs, from the shared alias list
(`schema/data/entities.json`)."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

ENTITIES_PATH = Path(__file__).resolve().parent.parent / "schema" / "data" / "entities.json"
COUNTRIES_PATH = ENTITIES_PATH.with_name("countries.json")

_SUFFIX = re.compile(r",?\s+(?:Inc|Incorporated|Corp|Corporation|Co|Company|plc|PLC|Ltd|Limited|"
                     r"LLC|S\.A|SA|AG|N\.V|NV|SE|Group|Holdings)\.?$")
_ODD = re.compile(r"[(),/]|\.(?:com|org|net)$")


@dataclass(frozen=True)
class Entity:
    scheme: str
    id: str
    name: str
    kind: str = ""


@dataclass(frozen=True)
class Mention:
    entity: Entity
    surface: str
    start: int
    end: int


def _capitalized(form: str) -> str:
    """NVIDIA as prose writes it: Nvidia."""
    return " ".join(w.capitalize() if len(w) > 4 and w.isupper() else w for w in form.split())


def _forms(name: str, aliases: Iterable[str]) -> List[str]:
    forms = [name, _SUFFIX.sub("", name), *aliases]
    forms += [_capitalized(f) for f in forms]
    return [f.strip() for f in dict.fromkeys(forms)
            if len(f.strip()) >= 3 and not _ODD.search(f) and f.strip() != f.strip().lower()]


def load_entities(path: Path = ENTITIES_PATH) -> List[Tuple[Entity, List[str]]]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return [(Entity(e["scheme"], e["id"], e["name"], e.get("kind", "")),
             _forms(e["name"], e.get("aliases", ()))) for e in raw.get("entities", [])]


class Linker:
    """Case-sensitive, whole-word matching of names and aliases; longest form wins."""

    def __init__(self, entities: Optional[List[Tuple[Entity, List[str]]]] = None):
        entities = load_entities() if entities is None else entities
        self.entities = [e for e, _ in entities]
        owners: Dict[str, Optional[Entity]] = {}
        for entity, forms in entities:
            for form in forms:
                owners[form] = entity if owners.get(form, entity) == entity else None
        # A form two entities share names neither.
        self._forms = {f: e for f, e in owners.items() if e is not None}
        self._rx = self._compile(self._forms)

    @staticmethod
    def _compile(forms: Iterable[str]) -> Optional[re.Pattern]:
        forms = sorted(forms, key=len, reverse=True)
        if not forms:
            return None
        return re.compile(r"(?<![\w&])(?:" + "|".join(map(re.escape, forms)) + r")(?![\w&])")

    def names(self, kind: str) -> List[str]:
        return [e.name for e in self.entities if e.kind == kind]

    def mentions(self, text: str) -> List[Mention]:
        if self._rx is None:
            return []
        return [Mention(self._forms[m.group(0)], m.group(0), m.start(), m.end())
                for m in self._rx.finditer(text)]


def load_countries(path: Path = COUNTRIES_PATH) -> Dict[str, str]:
    """Country names and aliases to ISO 3166 alpha-2 codes."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {form: c["iso2"] for c in raw.get("countries", [])
            for form in (c["name"], *c.get("aliases", ()))}


class PlaceFinder:
    """Countries a text names, in order of first mention."""

    def __init__(self, countries: Optional[Mapping[str, str]] = None):
        self.countries = dict(load_countries() if countries is None else countries)
        self._rx = Linker._compile(self.countries)

    def countries_in(self, text: str) -> List[str]:
        if self._rx is None:
            return []
        return list(dict.fromkeys(self.countries[m.group(0)] for m in self._rx.finditer(text)))

