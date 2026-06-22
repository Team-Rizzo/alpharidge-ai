"""
Persistent store for validator↔validator reward broadcasts.

We cache received broadcasts because:
- Validators may miss messages while offline.
- We apply rewards with a delay (e.g. apply epoch E-2).

Data model:
  last_seen_seq: {validator_hotkey: seq}
  by_epoch_by_sender: {epoch: {validator_hotkey: {uid: points}}}
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import bittensor as bt

from alpharidge_ai import config
from alpharidge_ai.utils import attestation_crypto as ac

# Defense-in-depth bounds on ingested broadcast data. A legit miner earns on the
# order of 1-50 points per epoch; anything above MAX_POINTS_PER_UID is fabricated
# (the MITM attacks injected ~65000). Legit broadcasts use seq == epoch, so a seq
# far from the stated epoch signals a rogue/poisoned broadcaster.
MAX_POINTS_PER_UID = 500
MAX_SEQ_EPOCH_SKEW = 100


def _default_path() -> Path:
    return Path(getattr(config, "BROADCAST_STATE_LOCATION", str(Path(__file__).resolve().parent.parent / ".broadcast_state.json")))


@dataclass
class RewardBroadcastStore:
    path: Path = field(default_factory=_default_path)
    keep_epochs: int = 3
    last_seen_seq: Dict[str, int] = field(default_factory=dict)
    by_epoch_by_sender: Dict[int, Dict[str, Dict[int, int]]] = field(default_factory=dict)
    merkle_by_epoch_by_sender: Dict[int, Dict[str, str]] = field(default_factory=dict)

    # ---------------------------------------------------------------------
    # Persistence
    # ---------------------------------------------------------------------
    def load(self) -> None:
        try:
            if not self.path.exists():
                return
            data = json.loads(self.path.read_text())
            self.last_seen_seq = {str(k): int(v) for k, v in (data.get("last_seen_seq") or {}).items()}
            raw = data.get("by_epoch_by_sender") or {}
            parsed: Dict[int, Dict[str, Dict[int, int]]] = {}
            for epoch_s, senders in raw.items():
                epoch = int(epoch_s)
                if not isinstance(senders, dict):
                    continue
                parsed[epoch] = {}
                for sender, uid_points in senders.items():
                    if not isinstance(uid_points, dict):
                        continue
                    parsed[epoch][str(sender)] = {int(uid): int(pts) for uid, pts in uid_points.items()}
            self.by_epoch_by_sender = parsed
            self.merkle_by_epoch_by_sender = {
                int(e): {str(s): str(m) for s, m in (senders or {}).items()}
                for e, senders in (data.get("merkle_by_epoch_by_sender") or {}).items()
            }
        except Exception as e:
            bt.logging.debug(f"[BROADCAST] Failed to load state {self.path}: {e}")

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "last_seen_seq": dict(self.last_seen_seq),
                "by_epoch_by_sender": {
                    str(epoch): {sender: {str(uid): int(pts) for uid, pts in uid_points.items()}
                                 for sender, uid_points in senders.items()}
                    for epoch, senders in self.by_epoch_by_sender.items()
                },
                "merkle_by_epoch_by_sender": {
                    str(e): dict(senders) for e, senders in self.merkle_by_epoch_by_sender.items()
                },
            }
            self.path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            bt.logging.debug(f"[BROADCAST] Failed to save state {self.path}: {e}")

    # ---------------------------------------------------------------------
    # Ingest
    # ---------------------------------------------------------------------
    def ingest(self, *, sender_hotkey: str, epoch: int, seq: int, uid_points: Dict[int, int]) -> Tuple[bool, str]:
        """
        Ingest a broadcast. Returns (accepted, reason).
        """
        sender = str(sender_hotkey)
        epoch_i = int(epoch)
        seq_i = int(seq)

        # Legit broadcasters set seq == epoch. A seq far from the stated epoch is a
        # rogue/poisoned broadcaster (the MITM attacks injected seq ~749000 to poison
        # last_seen_seq and deadlock all future legit broadcasts). Reject outright so
        # the poisoned seq never enters last_seen_seq.
        if abs(seq_i - epoch_i) > MAX_SEQ_EPOCH_SKEW:
            return False, f"seq_epoch_skew(seq={seq_i}, epoch={epoch_i})"

        last = int(self.last_seen_seq.get(sender, -1))
        if seq_i <= last:
            return False, f"duplicate_or_old_seq(last={last}, got={seq_i})"

        # Drop fabricated point values. A legit miner earns ~1-50 points/epoch; the
        # MITM attacks injected ~65000 to capture all incentive. Clamp rather than
        # reject the whole payload so a single bad UID can't suppress real ones.
        cleaned = {}
        for uid, pts in (uid_points or {}).items():
            p = int(pts)
            if p <= 0:
                continue
            if p > MAX_POINTS_PER_UID:
                bt.logging.warning(
                    f"[BROADCAST] Dropping out-of-bounds points from {sender[:12]}.. "
                    f"uid={int(uid)} points={p} (cap={MAX_POINTS_PER_UID})"
                )
                continue
            cleaned[int(uid)] = p
        if not cleaned:
            # Still advance last_seen_seq to prevent spam with empty payloads.
            self.last_seen_seq[sender] = seq_i
            return False, "empty_payload"

        # Store sender contribution for this epoch.
        self.by_epoch_by_sender.setdefault(epoch_i, {})[sender] = cleaned
        self.last_seen_seq[sender] = seq_i

        # Flush epochs from a stale numbering scheme before pruning.
        stale_epochs = [e for e in self.by_epoch_by_sender if e > epoch_i * 2]
        if stale_epochs:
            for e in stale_epochs:
                self.by_epoch_by_sender.pop(e, None)
            bt.logging.info(f"[BROADCAST] Flushed {len(stale_epochs)} stale epochs from old scheme")

        # Keep only the most recent N epochs.
        if self.keep_epochs > 0 and len(self.by_epoch_by_sender) > self.keep_epochs:
            for old_epoch in sorted(self.by_epoch_by_sender.keys())[:-self.keep_epochs]:
                self.by_epoch_by_sender.pop(old_epoch, None)

        return True, "accepted"

    def ingest_attestation(self, *, attestation: dict, signature: str, sender_hotkey: str,
                           hotkey_to_uid: Dict[str, int], pinned_pubkey: str,
                           seq: Optional[int] = None) -> Tuple[bool, str]:
        """Verify an API-signed attestation offline and ingest its per-miner points.
        The pinned pubkey is the root of trust; the API key is never taken over the wire.
        The wire `seq` is ignored; the signed `epoch` is the replay key."""
        sender = str(sender_hotkey)
        try:
            epoch_i = int(attestation["epoch"])
            per_miner = dict(attestation.get("perMinerPoints") or {})
            total = float(attestation.get("totalPoints") or 0.0)
            merkle_root = str(attestation.get("merkleRoot") or "")
            att_validator = str(attestation.get("validatorHotkey") or "")
        except Exception as e:
            return False, f"malformed_attestation({e})"

        if att_validator != sender:
            return False, "sender_mismatch"

        if not pinned_pubkey:
            return False, "no_pinned_pubkey"
        msg = ac.attestation_message(att_validator, epoch_i, per_miner, total, merkle_root)
        if not ac.verify_attestation(pinned_pubkey, msg, signature):
            return False, "bad_signature"

        # SECURITY: `seq` is not covered by the attestation signature, so a replayed
        # valid attestation could carry an inflated wire seq to poison last_seen_seq and
        # deadlock future legit broadcasts. Use the SIGNED epoch as the monotonic key;
        # the wire `seq` param is accepted for call-compatibility but deliberately ignored.
        seq_i = epoch_i
        last = int(self.last_seen_seq.get(sender, -1))
        if seq_i <= last:
            return False, f"duplicate_or_old_seq(last={last}, got={seq_i})"

        cleaned: Dict[int, int] = {}
        for hk, pts in per_miner.items():
            uid = hotkey_to_uid.get(hk)
            if uid is None:
                continue
            try:
                fv = float(pts)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(fv) or fv <= 0:
                continue
            p = int(round(fv))
            if p <= 0:
                continue
            if p > MAX_POINTS_PER_UID:
                bt.logging.warning(
                    f"[BROADCAST] Dropping out-of-bounds attestation points from {sender[:12]}.. "
                    f"hotkey={hk[:12]}.. points={p} (cap={MAX_POINTS_PER_UID})"
                )
                continue
            cleaned[int(uid)] = p

        if not cleaned:
            self.last_seen_seq[sender] = seq_i
            return False, "empty_payload"

        # Only retain a well-formed merkle root; junk would cause spurious deep-verify mismatches.
        if not (len(merkle_root) == 64 and all(c in "0123456789abcdef" for c in merkle_root)):
            merkle_root = ""

        self.by_epoch_by_sender.setdefault(epoch_i, {})[sender] = cleaned
        self.merkle_by_epoch_by_sender.setdefault(epoch_i, {})[sender] = merkle_root
        self.last_seen_seq[sender] = seq_i

        if self.keep_epochs > 0 and len(self.by_epoch_by_sender) > self.keep_epochs:
            for old_epoch in sorted(self.by_epoch_by_sender.keys())[:-self.keep_epochs]:
                self.by_epoch_by_sender.pop(old_epoch, None)
                self.merkle_by_epoch_by_sender.pop(old_epoch, None)

        return True, "accepted"

    def get_merkle_root(self, *, epoch: int, sender: str) -> Optional[str]:
        return (self.merkle_by_epoch_by_sender.get(int(epoch)) or {}).get(str(sender))

    # ---------------------------------------------------------------------
    # Aggregate
    # ---------------------------------------------------------------------
    def aggregate_epoch(self, epoch: int) -> Dict[int, int]:
        """
        Aggregate uid->points for a given epoch by summing across senders.
        """
        epoch_i = int(epoch)
        senders = self.by_epoch_by_sender.get(epoch_i) or {}
        agg: Dict[int, int] = {}
        for _sender, uid_points in senders.items():
            for uid, pts in uid_points.items():
                uid_i = int(uid)
                agg[uid_i] = agg.get(uid_i, 0) + int(pts)
        return agg

    # ---------------------------------------------------------------------
    # Remote reset helpers
    # ---------------------------------------------------------------------
    def flush_before_epoch(self, epoch: int) -> int:
        """Remove all broadcast data for epochs <= epoch and reset seq tracking. Returns count of epochs removed.

        last_seen_seq is always cleared, even when no epoch data was removed: a
        poisoned seq blocks all ingestion, leaving by_epoch_by_sender empty, so
        gating the clear on removed>0 made the deadlock unrecoverable via signal.
        """
        removed = 0
        for old_epoch in list(self.by_epoch_by_sender.keys()):
            if int(old_epoch) <= int(epoch):
                del self.by_epoch_by_sender[old_epoch]
                removed += 1
        self.last_seen_seq.clear()
        self.save()
        return removed

    def purge_hotkeys(self, hotkeys: list) -> None:
        """Remove all broadcast data referencing the given miner hotkeys (by UID lookup not needed — stored as UIDs)."""
        # Broadcasts store data as {epoch: {sender: {uid: pts}}} — we need a metagraph
        # to map hotkeys to UIDs. Instead, we accept UIDs directly OR hotkeys and
        # iterate through all entries removing matching UIDs.
        # For simplicity, accept a hotkey→uid mapping if available, otherwise just
        # remove any sender entries whose hotkey matches.
        hk_set = set(hotkeys)
        for epoch in list(self.by_epoch_by_sender.keys()):
            senders = self.by_epoch_by_sender[epoch]
            for sender in list(senders.keys()):
                if sender in hk_set:
                    del senders[sender]
            if not senders:
                del self.by_epoch_by_sender[epoch]
        self.save()


def route_reward_broadcast(*, store: "RewardBroadcastStore", sender_hotkey: str, epoch: int,
                           seq: int, uid_points: Dict[int, int], attestation: Optional[dict],
                           attestation_sig: Optional[str], hotkey_to_uid: Dict[str, int],
                           pinned_pubkey: str, blacklisted: Optional[set] = None,
                           enforce_signed: bool = False) -> Tuple[bool, str]:
    """Prefer the signed attestation (offline-verified). With enforce_signed=True (Phase 3),
    unsigned/legacy broadcasts are dropped; otherwise they fall back to legacy ingest during
    the grace window. Exactly ONE path runs per broadcast. Blacklisted senders are refused first."""
    if blacklisted and sender_hotkey in blacklisted:
        return False, "sender_blacklisted"
    if not (attestation and attestation_sig):
        if enforce_signed:
            return False, "unsigned_broadcast_rejected"
        return store.ingest(sender_hotkey=sender_hotkey, epoch=epoch, seq=seq, uid_points=uid_points)
    return store.ingest_attestation(
        attestation=attestation, signature=attestation_sig, sender_hotkey=sender_hotkey,
        hotkey_to_uid=hotkey_to_uid, pinned_pubkey=pinned_pubkey, seq=seq)


