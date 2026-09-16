"""Durable, consensus-safe reputation state.

Reputation is a fixed-weight mix of per-channel recency EMAs of graded scores, kept per
hotkey (see mechanism/channels.py). To keep both validators identical, per-article observations are broadcast and the
UNION (local + received) is applied at a delayed epoch close in a DETERMINISTIC order
(sort by article_id, then source) — EMA is order-dependent, so arrival order must not
matter. Mirrors the reward/penalty broadcast-store pattern (delayed application, keep a
few epochs, persist to JSON).

State is authoritative on disk; losing it resets all history, so it must be persisted and
backed up like the reward store.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

from alpharidge_ai import config
from alpharidge_ai.mechanism import channels as ch
from alpharidge_ai.validator import reputation as rep


def _default_path() -> Path:
    return Path(getattr(config, "REPUTATION_STATE_LOCATION",
                        str(Path(__file__).resolve().parent.parent / ".reputation_state.json")))


def _prior() -> float:
    """Cold-start reputation for an unseen hotkey. Read at the use-site (not bound at
    import) so the hourly served-config refresh takes effect without a restart."""
    try:
        return float(getattr(config, "REPUTATION_PRIOR", rep.PRIOR))
    except (TypeError, ValueError):
        return rep.PRIOR


# one observation: (article_id, graded, weight, channel code)
Obs = Tuple[int, float, float, int]

# Defense-in-depth bounds on ingested observations. A sender broadcasts once per epoch
# with seq == epoch, so a distant seq is a rogue or replayed payload; the volume caps
# bound how much EMA movement one sender can buy in a single epoch.
MAX_SEQ_EPOCH_SKEW = 100
MAX_OBS_PER_TARGET = 512
MAX_TARGETS_PER_SENDER = 1024


@dataclass
class ReputationStore:
    path: Path = field(default_factory=_default_path)
    keep_epochs: int = 4

    # durable per-hotkey state:
    #   hotkey -> {"r": reputation, "n": samples, "e": epoch, "c": {channel: {"r", "n"}}}
    # "r" is the weighted mean of the channels, kept current for readers.
    state: Dict[str, Dict] = field(default_factory=dict)
    channel_weights: Dict[str, float] = field(
        default_factory=lambda: dict(ch.DEFAULT_WEIGHTS))
    # pending observations: epoch -> sender_hotkey -> target_hotkey -> [Obs]
    obs: Dict[int, Dict[str, Dict[str, List[Obs]]]] = field(default_factory=dict)
    finalized: List[int] = field(default_factory=list)
    # highest accepted seq per sender
    last_seen_seq: Dict[str, int] = field(default_factory=dict)

    def load(self) -> None:
        try:
            if not self.path.exists():
                return
            data = json.loads(self.path.read_text())
            self.state = {str(k): self._load_entry(v)
                          for k, v in (data.get("state") or {}).items()}
            self.finalized = [int(e) for e in (data.get("finalized") or [])][-64:]
            self.last_seen_seq = {str(k): int(v)
                                  for k, v in (data.get("last_seen_seq") or {}).items()}
            raw = data.get("obs") or {}
            self.obs = {int(e): {s: {t: [self._as_obs(o) for o in lst] for t, lst in tgts.items()}
                                 for s, tgts in senders.items()}
                        for e, senders in raw.items()}
        except Exception:
            pass

    def save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({
            "state": self.state,
            "finalized": self.finalized[-64:],
            "obs": self.obs,
            "last_seen_seq": dict(self.last_seen_seq),
        }))
        tmp.replace(self.path)

    @staticmethod
    def _load_entry(v: Dict) -> Dict:
        entry = {"r": float(v["r"]), "n": int(v.get("n", 0))}
        if "e" in v:
            entry["e"] = int(v["e"])
        raw = v.get("c")
        if isinstance(raw, dict):
            entry["c"] = {str(name): {"r": float(c["r"]), "n": int(c.get("n", 0))}
                          for name, c in raw.items() if name in ch.CHANNELS}
        else:
            # State written before channels existed carries on as the legacy channel.
            entry["c"] = {ch.LEGACY: {"r": entry["r"], "n": entry["n"]}}
        return entry

    @staticmethod
    def _as_obs(o) -> Obs:
        code = int(float(o[3])) if len(o) > 3 else ch.CODES[ch.LEGACY]
        return (int(o[0]), float(o[1]), float(o[2]), code)

    def set_channel_weights(self, weights: Dict[str, float] = None) -> None:
        """Adopt published channel weights, and re-derive every reputation if they changed."""
        new = dict(ch.DEFAULT_WEIGHTS if weights is None else weights)
        if new == self.channel_weights:
            return
        self.channel_weights = new
        for st in self.state.values():
            st["r"] = ch.combine(st.get("c", {}), self.channel_weights, _prior())

    # ---- ingest ----
    def _add(self, epoch: int, sender: str, target: str, o) -> None:
        aid, g, w = int(o[0]), float(o[1]), float(o[2])
        if not (0.0 <= g <= 1.0) or not (0.0 < w <= 10.0):  # bounds guard
            return
        channel = ch.name_of(o[3]) if len(o) > 3 else ch.LEGACY
        if not channel:
            return
        self.obs.setdefault(epoch, {}).setdefault(sender, {}).setdefault(target, []).append(
            (aid, g, w, ch.CODES[channel]))

    def record_local(self, epoch: int, self_hotkey: str, target: str, article_id: int,
                     graded: float, weight: float, channel: str = ch.LEGACY) -> None:
        """Own observation — buffered for aggregation and for broadcast (via export)."""
        self._add(epoch, self_hotkey, target,
                  (article_id, graded, weight, ch.CODES[channel]))

    def ingest(self, sender: str, epoch: int, targets: Dict[str, List[Obs]],
               seq: int = None) -> Tuple[bool, str]:
        """A peer validator's observations for an epoch. Returns (accepted, reason)."""
        sender = str(sender)
        epoch_i = int(epoch)
        if epoch_i in self.finalized:
            return False, f"epoch_already_finalized({epoch_i})"

        if seq is not None:
            seq_i = int(seq)
            if abs(seq_i - epoch_i) > MAX_SEQ_EPOCH_SKEW:
                return False, f"seq_epoch_skew(seq={seq_i}, epoch={epoch_i})"
            last = int(self.last_seen_seq.get(sender, -1))
            if seq_i <= last:
                return False, f"duplicate_or_old_seq(last={last}, got={seq_i})"
            self.last_seen_seq[sender] = seq_i

        targets = targets or {}
        kept = 0
        # Truncate rather than reject: one oversized target must not suppress the rest.
        for target in sorted(targets)[:MAX_TARGETS_PER_SENDER]:
            for o in (targets.get(target) or [])[:MAX_OBS_PER_TARGET]:
                try:
                    self._add(epoch_i, sender, str(target), o)
                    kept += 1
                except (TypeError, ValueError, IndexError):
                    continue
        if not kept:
            return False, "empty_payload"
        return True, f"accepted({kept})"

    def export(self, epoch: int, self_hotkey: str) -> Dict[str, List[Obs]]:
        """Own observations for `epoch`, to broadcast to peers."""
        return dict((self.obs.get(epoch, {}) or {}).get(self_hotkey, {}))

    # ---- finalize (delayed) ----
    def finalize(self, epoch: int, alpha: float = None) -> None:
        """Apply the union of all senders' observations for `epoch` to the EMA in a
        deterministic order. Call for a delayed epoch (e.g. E-2) so broadcasts have settled."""
        if epoch in self.finalized or epoch not in self.obs:
            return
        alpha = rep.ALPHA if alpha is None else alpha
        # union per target, then order by (article_id, sender, channel)
        per_target: Dict[str, List[Tuple[int, str, int, float, float]]] = {}
        for sender, targets in self.obs[epoch].items():
            for target, lst in targets.items():
                # One observation per (article, sender, channel): keep the worst
                # score (ties: larger weight).
                best: Dict[Tuple[int, int], Tuple[float, float]] = {}
                for o in lst:
                    aid, g, w, code = self._as_obs(o)
                    key = (aid, code)
                    cur = best.get(key)
                    if cur is None or g < cur[0] or (g == cur[0] and w > cur[1]):
                        best[key] = (float(g), float(w))
                for (aid, code), (g, w) in best.items():
                    per_target.setdefault(target, []).append((aid, sender, code, g, w))
        for target, rows in per_target.items():
            rows.sort(key=lambda x: (x[0], x[1], x[2]))  # deterministic
            st = self.state.setdefault(target, {"r": _prior(), "n": 0, "c": {}})
            if "c" not in st:
                st["c"] = ({ch.LEGACY: {"r": float(st["r"]), "n": int(st["n"])}}
                           if int(st.get("n", 0)) > 0 else {})
            chans = st["c"]
            for _aid, _sender, code, g, w in rows:
                cur = chans.setdefault(ch.NAMES[code], {"r": _prior(), "n": 0})
                cur["r"] = rep.update(cur["r"], g, w, alpha)
                cur["n"] += 1
                st["n"] += 1
            st["r"] = ch.combine(chans, self.channel_weights, _prior())
            # Last epoch this hotkey was actually scored. Pruning needs to tell a hotkey
            # that has gone quiet from one that is merely absent from this batch.
            st["e"] = int(epoch)
        self.finalized.append(epoch)
        del self.obs[epoch]
        self._prune()
        self.save()

    def _prune(self) -> None:
        for e in sorted(self.obs)[:-self.keep_epochs]:
            del self.obs[e]
        self.finalized = self.finalized[-64:]

    # Roughly thirty days at 72 epochs a day.
    UNREGISTERED_GRACE_EPOCHS = 2160

    def prune_unregistered(self, registered, epoch: int,
                           grace_epochs: int = None) -> int:
        """Drop state for hotkeys that have left the metagraph and gone quiet.

        The store keeps every hotkey it has ever seen. On a subnet with churn that grows
        without bound, and it makes any population statistic taken from the store describe
        a field several times larger than the one being paid.

        Dropped after a grace period rather than on departure. A record that does not
        survive a brief absence is not a record, and the grace period is worth more than
        the disk space it costs. An entry with no recorded epoch is treated as current
        rather than ancient, so existing state is never dropped on the first pass.
        """
        allowed = {str(h) for h in (registered or ())}
        if not allowed:
            return 0
        grace = int(self.UNREGISTERED_GRACE_EPOCHS if grace_epochs is None else grace_epochs)
        gone = [hk for hk, st in self.state.items()
                if str(hk) not in allowed
                and int(epoch) - int(st.get("e", epoch)) > grace]
        for hk in gone:
            del self.state[hk]
        if gone:
            self.save()
        return len(gone)

    # ---- read ----
    def reputation(self, hotkey: str) -> float:
        return self.state.get(hotkey, {}).get("r", _prior())

    def samples(self, hotkey: str) -> int:
        return int(self.state.get(hotkey, {}).get("n", 0))

    def sender_observations(self, epoch: int, sender: str) -> Dict[str, List[Obs]]:
        """One sender's buffered observations for an epoch, by target."""
        return dict((self.obs.get(int(epoch), {}) or {}).get(str(sender), {}))

    def senders(self, epoch: int) -> List[str]:
        return sorted((self.obs.get(int(epoch), {}) or {}).keys())

    def snapshot(self) -> Dict[str, Dict]:
        """Per-hotkey {r, n, e, c} for telemetry / emission."""
        return {k: {**v, "c": {name: dict(c) for name, c in v.get("c", {}).items()}}
                for k, v in self.state.items()}
