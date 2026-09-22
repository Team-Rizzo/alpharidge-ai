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

import bittensor as bt
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

# Stands in for "no registration" on a record nobody currently holds.
UNHELD = -1

# A reconcile pass clears at most this many records; the oldest go first and the rest
# wait for the next pass, so a backlog (a validator that was down) drains on its own.
MAX_CLEAR_PER_PASS = 20
MAX_CLEAR_FRACTION = 0.05


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
    channel_alphas: Dict[str, float] = field(default_factory=dict)
    channel_defaults: Dict[str, float] = field(default_factory=dict)
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
        for field in ("u", "b"):
            if v.get(field) is not None:
                entry[field] = int(v[field])
        raw = v.get("c")
        if isinstance(raw, dict):
            # Channels this version does not know are kept as they are, so running an
            # older release against newer state does not erase their history.
            entry["c"] = {}
            for name, c in raw.items():
                try:
                    entry["c"][str(name)] = ReputationStore._load_channel(c)
                except (KeyError, TypeError, ValueError, AttributeError):
                    continue
        else:
            # State written before channels existed carries on as the legacy channel.
            entry["c"] = {ch.LEGACY: ReputationStore._settled(entry["r"], entry["n"])}
        return entry

    @staticmethod
    def _settled(r: float, n: int) -> Dict:
        return {"r": float(r), "n": int(n), "m": float(r), "d": 1.0}

    @staticmethod
    def _load_channel(c: Dict) -> Dict:
        if "m" not in c:
            return ReputationStore._settled(c["r"], c.get("n", 0))
        return {"r": float(c["r"]), "n": int(c.get("n", 0)),
                "m": float(c["m"]), "d": float(c["d"])}

    @staticmethod
    def _as_obs(o) -> Obs:
        code = int(float(o[3])) if len(o) > 3 else ch.CODES[ch.LEGACY]
        return (int(o[0]), float(o[1]), float(o[2]), code)

    def set_channel_weights(self, weights: Dict[str, float] = None,
                            alphas: Dict[str, float] = None,
                            defaults: Dict[str, float] = None) -> None:
        """Adopt published channel weights, steps and defaults, and re-derive every
        reputation if any of them changed."""
        new = dict(ch.DEFAULT_WEIGHTS if weights is None else weights)
        new_alphas = dict(alphas or {})
        new_defaults = dict(defaults or {})
        if (new == self.channel_weights and new_alphas == self.channel_alphas
                and new_defaults == self.channel_defaults):
            return
        self.channel_weights = new
        self.channel_alphas = new_alphas
        self.channel_defaults = new_defaults
        for st in self.state.values():
            st["r"] = self._combine(st.get("c", {}))

    def _combine(self, chans) -> float:
        return ch.combine(chans, self.channel_weights, _prior(), self.channel_alphas,
                          self.channel_defaults)

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

    @staticmethod
    def wire_payload(exported: Dict[str, List[Obs]]) -> Dict[str, List[List[float]]]:
        """Observations as the broadcast message carries them, channel code included."""
        return {t: [[float(x) for x in ReputationStore._as_obs(o)] for o in lst]
                for t, lst in (exported or {}).items()}

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
                st["c"] = ({ch.LEGACY: self._settled(st["r"], st["n"])}
                           if int(st.get("n", 0)) > 0 else {})
            chans = st["c"]
            for _aid, _sender, code, g, w in rows:
                name = ch.NAMES[code]
                step = self.channel_alphas.get(name, alpha)
                cur = chans.get(name)
                if cur is None:
                    # Bias-corrected: a new channel reads as the average of what it has
                    # seen, not as a blend with the prior.
                    cur = chans[name] = {"r": _prior(), "n": 0, "m": 0.0, "d": 0.0}
                elif "m" not in cur:
                    cur.update(self._settled(cur["r"], cur["n"]))
                cur["m"] = rep.update(cur["m"], g, w, step)
                cur["d"] = rep.update(cur["d"], 1.0, w, step)
                cur["r"] = cur["m"] / cur["d"] if cur["d"] > 0.0 else _prior()
                cur["n"] += 1
                st["n"] += 1
            st["r"] = self._combine(chans)
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

    # ---- identity ----

    def reconcile_identities(self, rows) -> Tuple[int, int]:
        """Bind each record to the registration that earned it.

        `rows` is (uid, hotkey, registration block) read from the chain. A record follows
        its registration: the same registration under a new hotkey keeps it, and a new
        registration starts empty, whichever hotkey holds it. Records with no binding yet
        are adopted by the registration currently holding them.

        Returns (cleared, moved). Both validators must pass the same rows.
        """
        rows = list(rows)
        cleared = moved = adopted = 0
        by_slot = {}
        for hk, st in self.state.items():
            if st.get("u") is not None and st.get("b") is not None:
                by_slot.setdefault((int(st["u"]), int(st["b"])), []).append(hk)
        for holders in by_slot.values():
            holders.sort()

        doomed = sorted(
            (int(self.state[hotkey]["b"]), str(hotkey), int(reg_block))
            for _, hotkey, reg_block in rows
            if hotkey in self.state
            and self.state[hotkey].get("b") is not None
            and int(self.state[hotkey]["b"]) != int(reg_block))
        limit = max(MAX_CLEAR_PER_PASS, int(MAX_CLEAR_FRACTION * len(self.state)))
        # A new registration is always later than the one a record is bound to. A row
        # that goes backwards is a read we cannot trust, so nothing is cleared on it.
        backwards = [d for d in doomed if d[0] != UNHELD and d[2] < d[0]]
        deferred = set()
        if backwards:
            bt.logging.error(
                f"[REPUTATION] {len(backwards)} row(s) carry a registration older than "
                f"the one on record; no record was cleared this pass")
            deferred = {hk for _, hk, _ in doomed}
        elif len(doomed) > limit:
            deferred = {hk for _, hk, _ in doomed[limit:]}
            bt.logging.warning(
                f"[REPUTATION] {len(doomed)} record(s) due to clear; clearing the oldest "
                f"{limit}, the rest next pass")

        for uid, hotkey, reg_block in rows:
            uid, hotkey, reg_block = int(uid), str(hotkey), int(reg_block)
            if hotkey in deferred:
                continue
            entry = self.state.get(hotkey)
            prior = next((p for p in by_slot.get((uid, reg_block), [])
                          if p != hotkey and p in self.state), None)
            if entry is not None:
                held = entry.get("b")
                if held is not None and int(held) != reg_block:
                    del self.state[hotkey]
                    cleared += 1
                    entry = None
                elif held is not None or prior is None:
                    if held is None:
                        adopted += 1
                    entry["u"], entry["b"] = uid, reg_block
                    continue

            # The registration's earlier record wins over one opened under the new hotkey
            # before this pass saw it.
            if prior is not None:
                carried = self.state.pop(prior)
                carried["u"], carried["b"] = uid, reg_block
                self.state[hotkey] = carried
                moved += 1

        # A record held by nobody is bound to no registration, so whichever one claims
        # its hotkey next starts empty.
        present = {str(hotkey) for _, hotkey, _ in rows}
        for hotkey, entry in self.state.items():
            if hotkey not in present and entry.get("b") is None:
                entry["u"], entry["b"] = UNHELD, UNHELD
                adopted += 1

        if cleared or moved or adopted:
            bt.logging.info(
                f"[REPUTATION] registrations reconciled: {cleared} cleared, {moved} "
                f"carried, {adopted} bound for the first time")
            self.save()
        return cleared, moved

    # ---- read ----
    def reputation(self, hotkey: str) -> float:
        st = self.state.get(hotkey)
        if st is None:
            return self._combine({}) if self.channel_defaults else _prior()
        return st.get("r", _prior())

    def channel_readiness(self, among=None) -> Dict[str, Tuple[int, int]]:
        """Per channel, (hotkeys past their warm-up, hotkeys counted)."""
        keys = [str(h) for h in among] if among is not None else list(self.state)
        out: Dict[str, Tuple[int, int]] = {}
        for name in ch.CHANNELS:
            ramp = ch.warmup(self.channel_alphas.get(name))
            warm = sum(1 for k in keys
                       if int(self.state.get(k, {}).get("c", {}).get(name, {}).get("n", 0))
                       >= ramp)
            out[name] = (warm, len(keys))
        return out

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
