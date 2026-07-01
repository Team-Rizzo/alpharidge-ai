# neurons/validator.py
# The MIT License (MIT)
# Copyright © 2023 Team Rizzo

"""
Validator entrypoint.
"""

# Baked-in launch env (set BEFORE bittensor/torch import so operators don't have to set them):
#   BT_NO_PARSE_CLI_ARGS   — bittensor 10.4 ignores CLI args (--netuid/--wallet) without this
#   CUBLAS_WORKSPACE_CONFIG — deterministic cuBLAS for cross-host consensus parity (must precede CUDA init)
import os
os.environ.setdefault("BT_NO_PARSE_CLI_ARGS", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import asyncio
import concurrent.futures
import copy
import gc
import time
from typing import List, Optional, Set

import bittensor as bt
from alpharidge_ai.base.validator import BaseValidatorNeuron
from alpharidge_ai.validator.forward import forward
from alpharidge_ai.validator.validation_client import ValidationClient
from alpharidge_ai.analyzer import setup_analyzer
from alpharidge_ai.analyzer import setup_news_analyzer
from alpharidge_ai.analyzer import setup_article_intelligence_analyzer
import alpharidge_ai.protocol
from alpharidge_ai import config
from alpharidge_ai.utils.api_models import TweetWithAuthor, CompletedTweetSubmission, TelegramMessageForScoring, CompletedTelegramMessageSubmission, TelegramMessageAnalysis, NewsArticleForScoring, CompletedNewsArticleSubmission
from alpharidge_ai.protocol import TweetBatch, TelegramBatch, ArticleBatch
from alpharidge_ai.utils.uids import get_random_uids, get_alive_uids
from alpharidge_ai.utils.liveness import LivenessRoster
from alpharidge_ai.utils.dispatch import coverage_depth_select
from alpharidge_ai.utils.dispatch_metrics import AdaptiveDispatchMetrics
from alpharidge_ai.utils.tweet_store import TweetStore
from alpharidge_ai.utils.telegram_store import TelegramStore
from alpharidge_ai.utils.article_store import ArticleStore
from alpharidge_ai.utils.reward import MinerReward
from alpharidge_ai.utils.penalty import MinerPenalty
from alpharidge_ai.validator.reward_broadcast_store import RewardBroadcastStore
from alpharidge_ai.validator.penalty_broadcast_store import PenaltyBroadcastStore
from alpharidge_ai.protocol import ValidatorRewards
from alpharidge_ai.protocol import ValidatorPenalties
from alpharidge_ai.analyzer.scoring import validate_miner_batch, validate_miner_telegram_batch, validate_miner_article_batch, validate_miner_article_intelligence_batch, classify_article_batch_failure
from alpharidge_ai.analyzer import setup_telegram_analyzer
from alpharidge_ai.utils.cooldown import MinerCooldownTracker
from alpharidge_ai.validator.verdict_payload import build_verdict_fields, collect_verdict_meta  # T5: verdict payload
class Validator(BaseValidatorNeuron):
    """
    Validator neuron for SN45.

    Clean flow:
    - Poll coordination API for tweets to process
    - Batch tweets and query miners over Bittensor (TweetBatch synapse)
    - Validate miner batches and mark tweets completed back to the API
    - Accumulate epoch rewards/penalties, broadcast to other validators, and set on-chain weights
    """

    def __init__(self, bt_config=None):
        # NOTE: this arg name must not shadow the imported `alpharidge_ai.config` module.
        super(Validator, self).__init__(config=bt_config)

        _vw = int(getattr(config, "VALIDATION_MAX_WORKERS", 2))
        self._validation_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=_vw,
            thread_name_prefix="validation_"
        )
        bt.logging.info(f"[INIT] Created validation executor with {_vw} workers")

        bt.logging.info("load_state()")
        self.load_state()

        # Initialize analyzer once (reused for all validations)
        bt.logging.info("[VALIDATION] Initializing analyzer...")
        self._analyzer = setup_analyzer()
        self._telegram_analyzer = setup_telegram_analyzer()
        self._news_analyzer = setup_news_analyzer()
        bt.logging.info("[VALIDATION] News analyzer initialized")

        try:
            self._article_intel_analyzer = setup_article_intelligence_analyzer()
            bt.logging.info("[VALIDATION] ArticleIntelligence analyzer initialized")
        except Exception as e:
            bt.logging.warning(f"[VALIDATION] ArticleIntelligence analyzer init failed, V1 only: {e}")
            self._article_intel_analyzer = None

        bt.logging.info("[VALIDATION] Analyzer initialized")

        # Initialize validation client
        self._validation_client = ValidationClient(validator=self, wallet=self.wallet)
        self._validation_task: Optional[asyncio.Task] = None
        self._liveness_task: Optional[asyncio.Task] = None
        self._tweet_store = TweetStore()
        self._telegram_store = TelegramStore()
        self._article_store = ArticleStore()
        # MinerReward / MinerPenalty expect a callable that returns the current block.
        # In Bittensor, `self.block` is an integer attribute (updated during sync), not a function.
        self._miner_reward = MinerReward(config.BLOCK_LENGTH, lambda: int(self.block))
        self._miner_penalty = MinerPenalty(config.BLOCK_LENGTH, lambda: int(self.block))
        # Rewards broadcast store: holds validator↔validator reward messages for delayed application.
        self._reward_broadcasts = RewardBroadcastStore()
        self._reward_broadcasts.load()
        # Transient per-item verdict metadata (resource_id -> {miner_signature, nonce,
        # validator_verdict, epoch}); populated during validation, drained at submission.
        self._verdict_meta = {}
        # Display-only penalty attribution buffer for the miner dashboard. DECOUPLED
        # from consensus: rows here are flushed best-effort to /diagnostics/penalty-detail
        # and never enter score_verdict / attestation / Merkle. Bounded so it can't grow
        # without limit if the API is unreachable (appends past the cap are dropped).
        self._penalty_detail_buffer = []
        self._penalty_detail_buffer_max = int(getattr(config, "PENALTY_DETAIL_BUFFER_MAX", 5000))
        # Penalties broadcast store: holds validator↔validator penalty messages for delayed application.
        self._penalty_broadcasts = PenaltyBroadcastStore()
        self._penalty_broadcasts.load()
        
        self._tweet_store.load_from_file()
        self._telegram_store.load_from_file()
        self._article_store.load_from_file()
        # Persisted stores expect a callable `block()`; pass a lambda (self.block is an int).
        self._miner_reward.load_from_file(block=lambda: int(self.block))
        self._miner_penalty.load_from_file(block=lambda: int(self.block))

        # Validator dispatches TweetBatch to miners (fire-and-forget).
        # Miners push analyzed TweetBatch back to this validator's axon when ready.
        self._miner_dispatch_semaphore = asyncio.Semaphore(
            max(1, int(getattr(config, "VALIDATOR_MINER_QUERY_CONCURRENCY", 8)))
        )
        self._pending_miner_tasks: Set[asyncio.Task] = set()
        self._max_pending_miner_tasks: int = int(
            getattr(config, "VALIDATOR_MAX_PENDING_MINER_TASKS", 256)
        )
        self._validating_tweet_ids: set = set()
        self._validating_message_ids: set = set()
        self._validating_article_ids: set = set()
        self._tweet_cooldown = MinerCooldownTracker()
        self._telegram_cooldown = MinerCooldownTracker()
        # Article tracker is the only adaptive one (RFC 2026-06-28); tweet/telegram
        # stay static. Behaves identically to static until ADAPTIVE_DISPATCH_ENABLED.
        self._article_cooldown = MinerCooldownTracker(adaptive=True)
        # Liveness roster (adaptive dispatch). Populated off the dispatch path;
        # only consulted for selection once ADAPTIVE_DISPATCH_ENABLED is on.
        self._liveness = LivenessRoster()
        # Per-cycle pilot metrics (adaptive dispatch).
        self._adaptive_metrics = AdaptiveDispatchMetrics()

    def resync_metagraph(self):
        super().resync_metagraph()
        # resync can fire during base __init__ (e.g. on a fast localnet) before the
        # cooldown trackers are created; skip pruning until they exist.
        if not hasattr(self, "_tweet_cooldown"):
            return
        active = set(self.metagraph.hotkeys)
        for tracker in (self._tweet_cooldown, self._telegram_cooldown, self._article_cooldown):
            tracker.prune(active)

    async def forward_tweets(self, synapse: alpharidge_ai.protocol.TweetBatch) -> alpharidge_ai.protocol.TweetBatch:
        """
        Axon handler for miner push-back of analyzed TweetBatch results.

        Validates store state synchronously (fast), then queues LLM validation
        as a background task so the axon returns immediately and the miner
        does not hit a 30s dendrite timeout.
        """
        miner_hotkey = synapse.dendrite.hotkey if synapse.dendrite else None
        if not miner_hotkey:
            return synapse

        bt.logging.info(f"[VALIDATION] Received TweetBatch with {len(synapse.tweet_batch)} tweet(s) from miner {miner_hotkey[:12]}..")

        sent_batch: List[TweetWithAuthor] = []
        for returned in synapse.tweet_batch:
            tid = str(getattr(returned, "id", ""))
            if not tid:
                continue
            if tid in self._validating_tweet_ids:
                bt.logging.info(
                    f"[VALIDATION] Dropping TweetBatch from {miner_hotkey[:12]}.. "
                    f"tweet {tid} already being validated (replay blocked)"
                )
                return synapse
            try:
                status = self._tweet_store.get_status(tid).value
                if status != "Processing":
                    bt.logging.info(
                        f"[VALIDATION] Dropping TweetBatch from {miner_hotkey[:12]}.. "
                        f"tweet {tid} status={status} (expected Processing)"
                    )
                    return synapse
                if self._tweet_store.get_hotkey(tid) != miner_hotkey:
                    bt.logging.info(
                        f"[VALIDATION] Dropping TweetBatch from {miner_hotkey[:12]}.. "
                        f"tweet {tid} hotkey mismatch"
                    )
                    return synapse
                sent_batch.append(self._tweet_store.get_tweet(tid))
            except Exception:
                return synapse

        if not sent_batch:
            return synapse

        # Lock these tweet IDs so replays are rejected while validation runs.
        batch_tids = {str(getattr(r, "id", "")) for r in synapse.tweet_batch if getattr(r, "id", "")}
        self._validating_tweet_ids.update(batch_tids)

        # Reset the timeout clock — the miner delivered results, we just need
        # time to grade them. Without this, slow LLM validation could trigger
        # a false timeout penalty even though results arrived on time.
        for returned in synapse.tweet_batch:
            tid = str(getattr(returned, "id", ""))
            if tid and tid in self._tweet_store._tweets:
                self._tweet_store._tweets[tid].start_time = time.time()

        # Queue validation as a background task so we return immediately.
        batch_copy = copy.deepcopy(synapse.tweet_batch)
        sent_batch_copy = copy.deepcopy(sent_batch)
        sigs = dict(getattr(synapse, "miner_signatures", {}) or {})
        ncs = dict(getattr(synapse, "nonces", {}) or {})

        async def _validate_and_release():
            try:
                await self._handle_miner_batch_response(batch_copy, miner_hotkey, sent_batch_copy, sigs, ncs)
            finally:
                self._validating_tweet_ids -= batch_tids

        task = asyncio.create_task(_validate_and_release())
        self._track_task(task)
        return synapse

    async def forward_telegram_messages(self, synapse: alpharidge_ai.protocol.TelegramBatch) -> alpharidge_ai.protocol.TelegramBatch:
        """
        Axon handler for miner push-back of analyzed TelegramBatch results.

        Validates store state synchronously (fast), then queues LLM validation
        as a background task so the axon returns immediately.
        """
        miner_hotkey = synapse.dendrite.hotkey if synapse.dendrite else None
        if not miner_hotkey:
            return synapse

        bt.logging.info(f"[VALIDATION] Received TelegramBatch with {len(synapse.message_batch)} message(s) from miner {miner_hotkey[:12]}..")

        sent_batch: List[TelegramMessageForScoring] = []
        for returned in synapse.message_batch:
            msg_id = str(getattr(returned, "id", ""))
            if not msg_id:
                continue
            if msg_id in self._validating_message_ids:
                bt.logging.info(
                    f"[VALIDATION] Dropping TelegramBatch from {miner_hotkey[:12]}.. "
                    f"message {msg_id} already being validated (replay blocked)"
                )
                return synapse
            try:
                status = self._telegram_store.get_status(msg_id).value
                if status != "Processing":
                    bt.logging.info(
                        f"[VALIDATION] Dropping TelegramBatch from {miner_hotkey[:12]}.. "
                        f"message {msg_id} status={status} (expected Processing)"
                    )
                    return synapse
                if self._telegram_store.get_hotkey(msg_id) != miner_hotkey:
                    bt.logging.info(
                        f"[VALIDATION] Dropping TelegramBatch from {miner_hotkey[:12]}.. "
                        f"message {msg_id} hotkey mismatch"
                    )
                    return synapse
                sent_batch.append(self._telegram_store.get_message(msg_id))
            except Exception:
                return synapse

        if not sent_batch:
            return synapse

        # Lock these message IDs so replays are rejected while validation runs.
        batch_mids = {str(getattr(r, "id", "")) for r in synapse.message_batch if getattr(r, "id", "")}
        self._validating_message_ids.update(batch_mids)

        # Reset the timeout clock — the miner delivered results, we just need
        # time to grade them.
        for returned in synapse.message_batch:
            msg_id = str(getattr(returned, "id", ""))
            if msg_id and msg_id in self._telegram_store._messages:
                self._telegram_store._messages[msg_id].start_time = time.time()

        # Queue validation as a background task so we return immediately.
        batch_copy = copy.deepcopy(synapse.message_batch)
        sent_batch_copy = copy.deepcopy(sent_batch)
        sigs = dict(getattr(synapse, "miner_signatures", {}) or {})
        ncs = dict(getattr(synapse, "nonces", {}) or {})

        async def _validate_and_release():
            try:
                await self._handle_telegram_miner_batch_response(batch_copy, miner_hotkey, sent_batch_copy, sigs, ncs)
            finally:
                self._validating_message_ids -= batch_mids

        task = asyncio.create_task(_validate_and_release())
        self._track_task(task)
        return synapse

    async def forward_articles(self, synapse: alpharidge_ai.protocol.ArticleBatch) -> alpharidge_ai.protocol.ArticleBatch:
        """
        Axon handler for miner push-back of analyzed ArticleBatch results.
        """
        miner_hotkey = synapse.dendrite.hotkey if synapse.dendrite else None
        if not miner_hotkey:
            return synapse

        # A push-back proves the miner is reachable — record liveness off the hot
        # path. In-memory only; never gates anything until the allocator lands.
        self._liveness.mark_seen(miner_hotkey)

        bt.logging.info(f"[VALIDATION] Received ArticleBatch with {len(synapse.article_batch)} article(s) from miner {miner_hotkey[:12]}..")

        sent_batch: List[NewsArticleForScoring] = []
        for returned in synapse.article_batch:
            aid = str(getattr(returned, "id", ""))
            if not aid:
                continue
            if aid in self._validating_article_ids:
                bt.logging.info(
                    f"[VALIDATION] Dropping ArticleBatch from {miner_hotkey[:12]}.. "
                    f"article {aid} already being validated (replay blocked)"
                )
                return synapse
            try:
                status = self._article_store.get_status(aid).value
                if status != "Processing":
                    bt.logging.info(
                        f"[VALIDATION] Dropping ArticleBatch from {miner_hotkey[:12]}.. "
                        f"article {aid} status={status} (expected Processing)"
                    )
                    return synapse
                if self._article_store.get_hotkey(aid) != miner_hotkey:
                    bt.logging.info(
                        f"[VALIDATION] Dropping ArticleBatch from {miner_hotkey[:12]}.. "
                        f"article {aid} hotkey mismatch"
                    )
                    return synapse
                sent_batch.append(self._article_store.get_article(aid))
            except Exception:
                return synapse

        if not sent_batch:
            return synapse

        batch_aids = {str(getattr(r, "id", "")) for r in synapse.article_batch if getattr(r, "id", "")}
        self._validating_article_ids.update(batch_aids)

        # Adaptive dispatch: capture the dispatch→push-back round-trip latency NOW,
        # before the reset below repurposes start_time for the validation clock. This
        # is miner-capacity latency only (it excludes the validator's own analyzer
        # time), which is what the congestion window must measure (RFC Component 2).
        _now = time.time()
        _starts = [
            self._article_store._articles[aid].start_time
            for aid in batch_aids
            if aid in self._article_store._articles and self._article_store._articles[aid].start_time
        ]
        pushback_latency_s = (_now - min(_starts)) if _starts else None

        for returned in synapse.article_batch:
            aid = str(getattr(returned, "id", ""))
            if aid and aid in self._article_store._articles:
                self._article_store._articles[aid].start_time = time.time()

        batch_copy = copy.deepcopy(synapse.article_batch)
        sent_batch_copy = copy.deepcopy(sent_batch)
        sigs = dict(getattr(synapse, "miner_signatures", {}) or {})
        ncs = dict(getattr(synapse, "nonces", {}) or {})

        async def _validate_and_release():
            try:
                await self._handle_article_miner_batch_response(batch_copy, miner_hotkey, sent_batch_copy, sigs, ncs, latency_s=pushback_latency_s)
            finally:
                self._validating_article_ids -= batch_aids

        task = asyncio.create_task(_validate_and_release())
        self._track_task(task)
        return synapse

    def _track_task(self, task: asyncio.Task) -> None:
        self._pending_miner_tasks.add(task)

        def _done(t: asyncio.Task) -> None:
            self._pending_miner_tasks.discard(t)
            try:
                exc = t.exception()
                if exc is not None:
                    bt.logging.debug(f"[VALIDATION] Miner dispatch task failed: {exc}")
            except asyncio.CancelledError:
                pass
            except Exception:
                pass

        task.add_done_callback(_done)

    async def _dispatch_miner_batch(self, miner_batch: List[TweetWithAuthor], uid: int) -> None:
        hotkey = None
        try:
            hotkey = self.metagraph.hotkeys[int(uid)]
        except Exception:
            pass
        if hotkey and not self._tweet_cooldown.try_acquire(hotkey):
            for tweet in miner_batch:
                try:
                    self._tweet_store.reset_to_unprocessed(tweet.id)
                except Exception:
                    pass
            return
        try:
            async with self._miner_dispatch_semaphore:
                await self._process_miner_batch(miner_batch, uid)
        finally:
            if hotkey:
                self._tweet_cooldown.release(hotkey)

    # ---- Display-only penalty attribution (decoupled from consensus) ----
    # The following two helpers feed self._penalty_detail_buffer, which is flushed
    # best-effort to /diagnostics/penalty-detail. None of this touches score_verdict,
    # attestation, Merkle, rewards, or penalty counts.

    # Map a discrepancy "reason" to the dashboard "cause". validator_classification_failed
    # is a validator-side failure (not the miner's fault), so it is not surfaced.
    _PENALTY_CAUSE_BY_REASON = {
        "classification_mismatch": "classification_mismatch",
        "missing_miner_classification": "missing_classification",
        "miner_needs_update": "needs_update",
    }

    def _buffer_penalty_detail(self, rows):
        """Append display-only attribution rows, bounded. Never raises."""
        try:
            if not rows:
                return
            buf = self._penalty_detail_buffer
            cap = self._penalty_detail_buffer_max
            for r in rows:
                if len(buf) >= cap:
                    # Drop oldest to stay bounded if the API has been unreachable.
                    del buf[0]
                buf.append(r)
        except Exception as e:
            bt.logging.debug(f"[PENALTY_DETAIL] buffer append skipped: {e}")

    def _penalty_rows_from_discrepancies(self, discrepancies, miner_hotkey, epoch, resource_type):
        """Build display-only penalty_detail rows from a rejected batch's discrepancies."""
        rows = []
        for disc in (discrepancies or []):
            try:
                cause = self._PENALTY_CAUSE_BY_REASON.get(disc.get("reason"))
                if cause is None:
                    continue  # skip validator-side / non-attributable reasons
                rid = disc.get("resource_id")
                if rid is None:
                    continue
                field_results = disc.get("field_results") or {}
                failed_fields = [k for k, ok in field_results.items() if not ok] or None
                preview = (disc.get("post_preview") or disc.get("message_preview")
                           or disc.get("article_preview"))
                rows.append({
                    "miner_hotkey": miner_hotkey,
                    "epoch": int(epoch),
                    "resource_type": resource_type,
                    "resource_id": str(rid),
                    "cause": cause,
                    "failed_fields": failed_fields,
                    "miner_values": disc.get("miner"),
                    "validator_values": disc.get("validator"),
                    "post_preview": preview,
                })
            except Exception as e:
                bt.logging.debug(f"[PENALTY_DETAIL] row build skipped: {e}")
                continue
        return rows

    async def _handle_miner_batch_response(
        self,
        tweet_batch: List[TweetWithAuthor],
        miner_hotkey: str,
        sent_batch: List[TweetWithAuthor],
        miner_signatures=None,
        nonces=None,
    ) -> bool:
        """
        Validate a miner's TweetBatch response and apply rewards/penalties exactly once per tweet.

        Args:
            tweet_batch: The batch returned by the miner.
            miner_hotkey: The miner's hotkey.
            sent_batch: The original batch sent to the miner (for size verification).

        Returns:
            True if batch accepted, False otherwise.
        """
        # Miner must return exactly what was sent (no cherry-picking).
        if len(tweet_batch) != len(sent_batch):
            bt.logging.warning(
                f"[VALIDATION] Batch size mismatch from miner {miner_hotkey} "
                f"sent {len(sent_batch)}, got {len(tweet_batch)}"
            )
            self._miner_penalty.add_penalty(miner_hotkey, 1)
            for tweet in sent_batch:
                try:
                    self._tweet_store.reset_to_unprocessed(tweet.id)
                except Exception:
                    pass
            return False

        # Validate by re-running analyzer on sampled posts.
        loop = asyncio.get_running_loop()
        is_valid, validation_result = await loop.run_in_executor(
            self._validation_executor,
            validate_miner_batch, tweet_batch, self._analyzer, 1
        )
        if not is_valid:
            # Log detailed rejection reason
            discrepancies = validation_result.get("discrepancies", [])
            match_rate = validation_result.get("match_rate", 0.0)
            bt.logging.warning(
                f"[VALIDATION] Batch validation FAILED for miner {miner_hotkey} "
                f"match_rate={match_rate:.1%}, discrepancies={len(discrepancies)}"
            )
            for disc in discrepancies:
                reason = disc.get("reason", "unknown")
                preview = disc.get("post_preview", "")
                if reason == "classification_mismatch":
                    field_results = disc.get("field_results", {})
                    failed_fields = [k for k, v in field_results.items() if not v]
                    miner_vals = disc.get("miner", {})
                    validator_vals = disc.get("validator", {})
                    # Log each failed field with miner vs validator values
                    field_comparisons = []
                    for f in failed_fields:
                        m = miner_vals.get(f, "?")
                        v = validator_vals.get(f, "?")
                        field_comparisons.append(f"{f}(m={m}|v={v})")
                    bt.logging.warning(
                        f"[VALIDATION] Mismatch for {miner_hotkey}: {', '.join(field_comparisons)} | preview={preview[:100]}"
                    )
                else:
                    bt.logging.warning(f"[VALIDATION] Rejection for {miner_hotkey}: reason={reason}, preview={preview[:100]}")
            
            current_epoch = self._miner_reward._get_current_epoch()
            # V6: do NOT write invalid entries into _verdict_meta — invalid items are
            # reset_to_unprocessed and never submitted, so those entries would leak.
            # V2: record display-only attribution instead (decoupled from consensus).
            self._buffer_penalty_detail(
                self._penalty_rows_from_discrepancies(discrepancies, miner_hotkey, current_epoch, "tweet"))
            self._miner_penalty.add_penalty(miner_hotkey, 1)
            for tweet in tweet_batch:
                try:
                    self._tweet_store.reset_to_unprocessed(tweet.id)
                except Exception:
                    pass
            return False

        bt.logging.info(f"[VALIDATION] Batch validation PASSED for miner {miner_hotkey}")
        self._tweet_cooldown.record_success(miner_hotkey)
        # Batch accepted: persist analyzed tweets, mark processed, and reward once per tweet.
        for tweet in tweet_batch:
            # Ensure store has the analyzed tweet for API submission.
            try:
                self._tweet_store.update_tweet(tweet.id, tweet)
            except Exception:
                # If missing, add it.
                self._tweet_store.add_tweet(tweet, tweet_id=tweet.id, hotkey=miner_hotkey, set_as_processing=False, overwrite=True)

            try:
                self._tweet_store.set_processed(tweet.id)
            except Exception:
                pass

            # Idempotent reward: only reward once per tweet_id.
            if not self._tweet_store.is_rewarded(tweet.id):
                self._miner_reward.add_reward(miner_hotkey, 1)
                try:
                    self._tweet_store.mark_rewarded(tweet.id)
                except Exception:
                    pass

        current_epoch = self._miner_reward._get_current_epoch()
        self._verdict_meta.update(
            collect_verdict_meta(tweet_batch, miner_signatures, nonces, "valid", current_epoch))
        return True

    async def _handle_telegram_miner_batch_response(
        self,
        message_batch: List[TelegramMessageForScoring],
        miner_hotkey: str,
        sent_batch: List[TelegramMessageForScoring],
        miner_signatures=None,
        nonces=None,
    ) -> bool:
        """
        Validate a miner's TelegramBatch response and apply rewards/penalties exactly once per message.

        Args:
            message_batch: The batch returned by the miner.
            miner_hotkey: The miner's hotkey.
            sent_batch: The original batch sent to the miner (for size verification).

        Returns:
            True if batch accepted, False otherwise.
        """
        # Miner must return exactly what was sent (no cherry-picking).
        if len(message_batch) != len(sent_batch):
            bt.logging.warning(
                f"[VALIDATION] Telegram batch size mismatch from miner {miner_hotkey} "
                f"sent {len(sent_batch)}, got {len(message_batch)}"
            )
            self._miner_penalty.add_penalty(miner_hotkey, 1)
            for msg in sent_batch:
                try:
                    self._telegram_store.reset_to_unprocessed(msg.id)
                except Exception:
                    pass
            return False

        # Validate by re-running analyzer on sampled messages.
        loop = asyncio.get_running_loop()
        is_valid, validation_result = await loop.run_in_executor(
            self._validation_executor,
            validate_miner_telegram_batch, message_batch, self._telegram_analyzer, 1
        )
        if not is_valid:
            # Log detailed rejection reason
            discrepancies = validation_result.get("discrepancies", [])
            match_rate = validation_result.get("match_rate", 0.0)
            bt.logging.warning(
                f"[VALIDATION] Telegram batch validation FAILED for miner {miner_hotkey} "
                f"match_rate={match_rate:.1%}, discrepancies={len(discrepancies)}"
            )
            for disc in discrepancies:
                reason = disc.get("reason", "unknown")
                preview = disc.get("message_preview", "")
                if reason == "classification_mismatch":
                    field_results = disc.get("field_results", {})
                    failed_fields = [k for k, v in field_results.items() if not v]
                    miner_vals = disc.get("miner", {})
                    validator_vals = disc.get("validator", {})
                    # Log each failed field with miner vs validator values
                    field_comparisons = []
                    for f in failed_fields:
                        m = miner_vals.get(f, "?")
                        v = validator_vals.get(f, "?")
                        field_comparisons.append(f"{f}(m={m}|v={v})")
                    bt.logging.warning(
                        f"[VALIDATION] Telegram mismatch for {miner_hotkey}: {', '.join(field_comparisons)} | preview={preview[:100]}"
                    )
                else:
                    bt.logging.warning(f"[VALIDATION] Telegram rejection for {miner_hotkey}: reason={reason}, preview={preview[:100]}")
            
            current_epoch = self._miner_reward._get_current_epoch()
            # V6/V2: see tweet path — drop the leaking invalid _verdict_meta write and
            # record decoupled display-only attribution instead.
            self._buffer_penalty_detail(
                self._penalty_rows_from_discrepancies(discrepancies, miner_hotkey, current_epoch, "telegram"))
            self._miner_penalty.add_penalty(miner_hotkey, 1)
            for msg in message_batch:
                try:
                    self._telegram_store.reset_to_unprocessed(msg.id)
                except Exception:
                    pass
            return False

        bt.logging.info(f"[VALIDATION] Telegram batch validation PASSED for miner {miner_hotkey}")
        self._telegram_cooldown.record_success(miner_hotkey)
        # Batch accepted: persist analyzed messages, mark processed, and reward once per message.
        for msg in message_batch:
            # Ensure store has the analyzed message for API submission.
            try:
                self._telegram_store.update_message(msg.id, msg)
            except Exception:
                # If missing, add it.
                self._telegram_store.add_message(msg, message_id=msg.id, hotkey=miner_hotkey, set_as_processing=False, overwrite=True)

            try:
                self._telegram_store.set_processed(msg.id)
            except Exception:
                pass

            # Idempotent reward: only reward once per message_id.
            if not self._telegram_store.is_rewarded(msg.id):
                self._miner_reward.add_reward(miner_hotkey, 1)
                try:
                    self._telegram_store.mark_rewarded(msg.id)
                except Exception:
                    pass

        current_epoch = self._miner_reward._get_current_epoch()
        self._verdict_meta.update(
            collect_verdict_meta(message_batch, miner_signatures, nonces, "valid", current_epoch))
        return True

    async def _handle_article_miner_batch_response(
        self,
        article_batch: List[NewsArticleForScoring],
        miner_hotkey: str,
        sent_batch: List[NewsArticleForScoring],
        miner_signatures=None,
        nonces=None,
        latency_s: float = None,
    ) -> bool:
        adaptive = getattr(config, "ADAPTIVE_DISPATCH_ENABLED", False)
        if len(article_batch) != len(sent_batch):
            bt.logging.warning(
                f"[VALIDATION] Article batch size mismatch from miner {miner_hotkey} "
                f"sent {len(sent_batch)}, got {len(article_batch)}"
            )
            self._miner_penalty.add_penalty(miner_hotkey, 1)
            if adaptive:
                self._article_cooldown.record_invalid(miner_hotkey)
                self._adaptive_metrics.incr("invalid")
            for article in sent_batch:
                try:
                    self._article_store.reset_to_unprocessed(article.id)
                except Exception:
                    pass
            return False

        loop = asyncio.get_running_loop()

        # Try V2 validation if miner submitted analysis_data
        has_v2 = any(
            getattr(a.analysis, "analysis_data", None)
            for a in article_batch if a.analysis
        )
        if has_v2 and self._article_intel_analyzer is not None:
            is_valid, validation_result = await loop.run_in_executor(
                self._validation_executor,
                validate_miner_article_intelligence_batch,
                article_batch, self._article_intel_analyzer, 1,
            )
        else:
            is_valid, validation_result = await loop.run_in_executor(
                self._validation_executor,
                validate_miner_article_batch, article_batch, self._news_analyzer, 1,
            )
        if not is_valid:
            discrepancies = validation_result.get("discrepancies", [])
            match_rate = validation_result.get("match_rate", 0.0)
            bt.logging.warning(
                f"[VALIDATION] Article batch validation FAILED for miner {miner_hotkey} "
                f"match_rate={match_rate:.1%}, discrepancies={len(discrepancies)}"
            )
            for disc in discrepancies:
                reason = disc.get("reason", "unknown")
                preview = disc.get("article_preview", "")
                if reason == "classification_mismatch":
                    field_results = disc.get("field_results", {})
                    failed_fields = [k for k, v in field_results.items() if not v]
                    miner_vals = disc.get("miner", {})
                    validator_vals = disc.get("validator", {})
                    field_comparisons = []
                    for f in failed_fields:
                        m = miner_vals.get(f, "?")
                        v = validator_vals.get(f, "?")
                        field_comparisons.append(f"{f}(m={m}|v={v})")
                    bt.logging.warning(
                        f"[VALIDATION] Article mismatch for {miner_hotkey}: {', '.join(field_comparisons)} | preview={preview[:100]}"
                    )
                else:
                    bt.logging.warning(f"[VALIDATION] Article rejection for {miner_hotkey}: reason={reason}, preview={preview[:100]}")

            current_epoch = self._miner_reward._get_current_epoch()
            # Missing/incomplete-analysis is a CAPACITY signal (the miner hasn't caught up
            # with the burst we dispatched), not cheating — split it out like a timeout
            # rather than scoring it as an integrity failure. Gated separately so it can
            # be piloted / rolled back on its own; off = legacy "penalize everything".
            missing_split = adaptive and getattr(config, "ADAPTIVE_MISSING_ANALYSIS_SPLIT_ENABLED", False)
            failure_class = classify_article_batch_failure(discrepancies)

            if missing_split and failure_class != "integrity":
                if failure_class == "validator_side":
                    # Our own analyzer failed — the miner did nothing wrong and isn't even
                    # overloaded: no penalty, no window change, just retry.
                    bt.logging.info(
                        f"[VALIDATION] validator-side analysis failure for {miner_hotkey} — no penalty, retrying")
                else:
                    # Capacity: back off the dispatch window (so we stop dumping work it
                    # can't drain), but no integrity penalty / no emission-gate hit.
                    bt.logging.info(
                        f"[VALIDATION] incomplete analysis from {miner_hotkey} (capacity, not integrity) "
                        f"— backing off window, no penalty")
                    self._article_cooldown.record_invalid(miner_hotkey)
                    self._adaptive_metrics.incr("incomplete")
            else:
                # Genuine integrity failure (or the split is disabled) — unchanged:
                # display-only attribution + integrity penalty + window shrink.
                self._buffer_penalty_detail(
                    self._penalty_rows_from_discrepancies(discrepancies, miner_hotkey, current_epoch, "article"))
                self._miner_penalty.add_penalty(miner_hotkey, 1)
                if adaptive:
                    self._article_cooldown.record_invalid(miner_hotkey)
                    self._adaptive_metrics.incr("invalid")

            for article in article_batch:
                try:
                    self._article_store.reset_to_unprocessed(article.id)
                except Exception:
                    pass
            return False

        bt.logging.info(f"[VALIDATION] Article batch validation PASSED for miner {miner_hotkey}")
        self._article_cooldown.record_success(miner_hotkey)
        # Adaptive: grow the window if the round-trip was comfortably on-time, else
        # freeze (objective 8 — find capacity without ramping into a timeout).
        if adaptive:
            self._article_cooldown.record_timely_valid(miner_hotkey, latency_s)
            self._adaptive_metrics.incr("valid")
            self._adaptive_metrics.mark_scored(miner_hotkey)
        for article in article_batch:
            try:
                self._article_store.update_article(article.id, article)
            except Exception:
                self._article_store.add_article(article, article_id=article.id, hotkey=miner_hotkey, set_as_processing=False, overwrite=True)

            try:
                self._article_store.set_processed(article.id)
            except Exception:
                pass

            if not self._article_store.is_rewarded(article.id):
                content_len = len(article.content or "") if article.content else 0
                if content_len >= 2000:
                    weight = 3
                elif content_len >= 500:
                    weight = 2
                else:
                    weight = 1
                self._miner_reward.add_reward(miner_hotkey, weight)
                try:
                    self._article_store.mark_rewarded(article.id)
                except Exception:
                    pass

        current_epoch = self._miner_reward._get_current_epoch()
        self._verdict_meta.update(
            collect_verdict_meta(article_batch, miner_signatures, nonces, "valid", current_epoch))
        return True

    async def _on_tweets(self, tweets: List[TweetWithAuthor]):
        """
        Process multiple tweets in batch (sequentially).
        
        Args:
            tweets: List of tweets
        """
        if not tweets:
            return
        
        bt.logging.info(f"[VALIDATION] Processing {len(tweets)} tweets in batch")
        for tweet in tweets:
            # Preserve existing store entries (avoid losing processed/submitted/rewarded flags).
            self._tweet_store.add_tweet(tweet, set_as_processing=False, overwrite=False)
        miner_batches = []
        for i in range(0, len(tweets), config.MINER_BATCH_SIZE):
            miner_batches.append(tweets[i:i + config.MINER_BATCH_SIZE])
        # Exclude ourselves and miners on cooldown from dispatch selection.
        cooled_hotkeys = self._tweet_cooldown.get_cooled_down_hotkeys()
        cooled_uids = [
            uid for uid in range(self.metagraph.n.item())
            if self.metagraph.hotkeys[uid] in cooled_hotkeys
        ]
        exclude = [int(self.uid)] + cooled_uids
        uids = list(get_random_uids(self, k=len(miner_batches), exclude=exclude))
        tracked, on_cd = self._tweet_cooldown.stats()
        if on_cd > 0:
            available = len(uids)
            bt.logging.debug(f"[COOLDOWN/tweet] {on_cd} miners on cooldown, {available} available for dispatch")

        for miner_batch, uid in zip(miner_batches, uids):
            if len(self._pending_miner_tasks) >= self._max_pending_miner_tasks:
                bt.logging.warning(
                    f"[VALIDATION] Too many pending miner dispatch tasks ({len(self._pending_miner_tasks)}); "
                    f"skipping scheduling remaining batches this tick."
                )
                break
            task = asyncio.create_task(self._dispatch_miner_batch(miner_batch, int(uid)))
            self._track_task(task)

    async def _on_telegram_messages(self, messages: List[TelegramMessageForScoring]):
        """
        Process multiple telegram messages in batch.
        
        Args:
            messages: List of TelegramMessageForScoring
        """
        if not messages:
            return
        
        bt.logging.info(f"[VALIDATION] Processing {len(messages)} telegram messages in batch")
        for msg in messages:
            # Preserve existing store entries (avoid losing processed/submitted/rewarded flags).
            self._telegram_store.add_message(msg, set_as_processing=False, overwrite=False)
        miner_batches = []
        for i in range(0, len(messages), config.MINER_BATCH_SIZE):
            miner_batches.append(messages[i:i + config.MINER_BATCH_SIZE])
        cooled_hotkeys = self._telegram_cooldown.get_cooled_down_hotkeys()
        cooled_uids = [
            uid for uid in range(self.metagraph.n.item())
            if self.metagraph.hotkeys[uid] in cooled_hotkeys
        ]
        exclude = [int(self.uid)] + cooled_uids
        uids = list(get_random_uids(self, k=len(miner_batches), exclude=exclude))

        for miner_batch, uid in zip(miner_batches, uids):
            if len(self._pending_miner_tasks) >= self._max_pending_miner_tasks:
                bt.logging.warning(
                    f"[VALIDATION] Too many pending miner dispatch tasks ({len(self._pending_miner_tasks)}); "
                    f"skipping scheduling remaining telegram batches this tick."
                )
                break
            task = asyncio.create_task(self._dispatch_telegram_miner_batch(miner_batch, int(uid)))
            self._track_task(task)

    async def _on_articles(self, articles: List[NewsArticleForScoring]):
        if not articles:
            return

        bt.logging.info(f"[VALIDATION] Processing {len(articles)} articles in batch")
        for article in articles:
            self._article_store.add_article(article, set_as_processing=False, overwrite=False)
        miner_batches = []
        for i in range(0, len(articles), config.MINER_BATCH_SIZE):
            miner_batches.append(articles[i:i + config.MINER_BATCH_SIZE])
        cooled_hotkeys = self._article_cooldown.get_cooled_down_hotkeys()
        cooled_uids = [
            uid for uid in range(self.metagraph.n.item())
            if self.metagraph.hotkeys[uid] in cooled_hotkeys
        ]
        exclude = [int(self.uid)] + cooled_uids
        targets = self._select_article_targets(miner_batches, exclude)

        adaptive = getattr(config, "ADAPTIVE_DISPATCH_ENABLED", False)
        epoch = self._current_epoch() if adaptive else 0
        dispatched_uids_this_tick = set()
        for uid, miner_batch in targets:
            if len(self._pending_miner_tasks) >= self._max_pending_miner_tasks:
                bt.logging.warning(
                    f"[VALIDATION] Too many pending miner dispatch tasks ({len(self._pending_miner_tasks)}); "
                    f"skipping scheduling remaining article batches this tick."
                )
                break
            # A 2nd+ batch to the same miner within one tick is a depth dispatch
            # (the coverage floor only ever assigns one). Count it so the metrics
            # line shows how much of the volume is depth vs coverage.
            if adaptive:
                if int(uid) in dispatched_uids_this_tick:
                    self._adaptive_metrics.incr("depth_dispatched")
                else:
                    dispatched_uids_this_tick.add(int(uid))
            task = asyncio.create_task(self._dispatch_article_miner_batch(miner_batch, int(uid)))
            self._track_task(task)
            # Mark covered on actual dispatch (not at allocation time): if the pending-cap
            # break above drops a coverage assignment, the miner must NOT be recorded as
            # covered for this epoch without having been sent work.
            if adaptive:
                try:
                    self._article_cooldown.mark_covered(self.metagraph.hotkeys[int(uid)], epoch)
                except Exception:
                    pass

    def _reconcile_article_inflight(self):
        """Rebuild per-miner in-flight from the article store's PROCESSING set, so a
        missed or duplicated completion event can never strand the window. PROCESSING
        is per-article; the window is per-batch, so convert with
        ceil(articles / batch_size). This is the sole source of truth for in-flight
        under adaptive dispatch (the dispatch coroutine no longer releases at the ack)."""
        batch_size = max(1, int(getattr(config, "MINER_BATCH_SIZE", 20)))
        article_counts = {}
        for item in self._article_store.get_processing_articles():
            hk = getattr(item, "hotkey", None)
            if hk:
                article_counts[hk] = article_counts.get(hk, 0) + 1
        batch_counts = {hk: -(-c // batch_size) for hk, c in article_counts.items()}
        self._article_cooldown.reconcile_inflight(batch_counts)

    def _current_epoch(self) -> int:
        try:
            return int(self.block) // int(getattr(config, "BLOCK_LENGTH", 100))
        except Exception:
            return 0

    def _build_dispatch_status(self) -> list:
        """Per-miner adaptive-dispatch status snapshot for the dashboard diagnostics
        flush (display-only, consensus-decoupled). Covers every currently-live miner
        plus any miner we hold cooldown/window state for."""
        hotkeys = list(self.metagraph.hotkeys)
        hk_to_uid = {hk: u for u, hk in enumerate(hotkeys)}
        ct = self._article_cooldown.snapshot()
        live_hks = {hotkeys[u] for u in self._liveness.live_uids(self.metagraph)}
        w_min = float(getattr(config, "DISPATCH_WINDOW_MIN", 1))
        rows = []
        for hk in (set(ct) | live_hks):
            st = ct.get(hk, {})
            rows.append({
                "hotkey": hk,
                "uid": int(hk_to_uid.get(hk, -1)),
                "alive": bool(self._liveness.is_alive(hk)),
                "window": float(st.get("window", w_min)),
                "inflight": int(st.get("inflight", 0)),
                "consec_to": int(st.get("consec_to", 0)),
                "covered_epoch": int(st.get("covered_epoch", -1)),
                "on_cooldown": bool(st.get("on_cooldown", False)),
                "cooldown_remaining_s": int(st.get("cooldown_remaining_s", 0)),
            })
        return rows

    def _select_article_targets(self, miner_batches, exclude):
        """
        Choose (uid, batch) dispatch targets.

        Flag off: unchanged random selection. Flag on: coverage-then-depth over the
        live roster (see utils/dispatch.coverage_depth_select). Read-only on the
        tracker here — the real per-miner reservation stays in
        _dispatch_article_miner_batch.try_acquire, so a pending-cap truncation cannot
        leak a reserved slot.
        """
        n_batches = len(miner_batches)
        if not getattr(config, "ADAPTIVE_DISPATCH_ENABLED", False):
            uids = list(get_random_uids(self, k=n_batches, exclude=exclude))
            return [(int(u), b) for b, u in zip(miner_batches, uids)]

        # Sync in-flight to ground truth before allocating (leak-proof; see method).
        self._reconcile_article_inflight()

        hotkeys = list(self.metagraph.hotkeys)
        blacklisted = getattr(config, "BLACKLISTED_MINER_HOTKEYS", set()) or set()
        exclude_set = {int(u) for u in exclude}
        live = [
            u for u in self._liveness.live_uids(self.metagraph)
            if u not in exclude_set and 0 <= u < len(hotkeys) and hotkeys[u] not in blacklisted
        ]
        if not live:
            bt.logging.warning("[DISPATCH] adaptive dispatch on but live roster is empty this tick; no targets")
            return []

        # Anti-monopoly cap, recomputed per tick: cap_pct of the validator's total
        # in-flight send budget (a stable, exogenous quantity — NOT the sum of
        # windows, which would feed back and run away). Grows when concurrency is
        # raised in Component 4. With the default concurrency of 8 this keeps windows
        # at ~1 (coverage only) until concurrency is deliberately raised.
        cap_pct = float(getattr(config, "DISPATCH_WINDOW_CAP_PCT", 0.15))
        w_min = float(getattr(config, "DISPATCH_WINDOW_MIN", 1))
        budget = float(getattr(config, "VALIDATOR_MINER_QUERY_CONCURRENCY", 8))
        self._article_cooldown.set_cap(max(w_min, cap_pct * budget))

        epoch = self._current_epoch()
        assignments = coverage_depth_select(live, hotkeys, self._article_cooldown, epoch, n_batches)

        n_assigned = len(assignments)
        if n_assigned < n_batches:
            bt.logging.info(
                f"[DISPATCH] adaptive: {n_batches - n_assigned}/{n_batches} batch(es) unassigned this "
                f"tick (live windows full); they stay unprocessed and retry next tick."
            )
        else:
            distinct = len({u for u, _ in assignments})
            bt.logging.info(f"[DISPATCH] adaptive: assigned {n_assigned} batch(es) across {distinct} live miner(s)")
        return [(uid, miner_batches[bi]) for uid, bi in assignments]

    async def _dispatch_telegram_miner_batch(self, miner_batch: List[TelegramMessageForScoring], uid: int) -> None:
        hotkey = None
        try:
            hotkey = self.metagraph.hotkeys[int(uid)]
        except Exception:
            pass
        if hotkey and not self._telegram_cooldown.try_acquire(hotkey):
            for msg in miner_batch:
                try:
                    self._telegram_store.reset_to_unprocessed(msg.id)
                except Exception:
                    pass
            return
        try:
            async with self._miner_dispatch_semaphore:
                await self._process_telegram_miner_batch(miner_batch, uid)
        finally:
            if hotkey:
                self._telegram_cooldown.release(hotkey)

    async def _dispatch_article_miner_batch(self, miner_batch: List[NewsArticleForScoring], uid: int) -> None:
        hotkey = None
        try:
            hotkey = self.metagraph.hotkeys[int(uid)]
        except Exception:
            pass
        if hotkey and not self._article_cooldown.try_acquire(hotkey):
            for article in miner_batch:
                try:
                    self._article_store.reset_to_unprocessed(article.id)
                except Exception:
                    pass
            return
        try:
            async with self._miner_dispatch_semaphore:
                await self._process_article_miner_batch(miner_batch, uid)
        finally:
            # Static path releases at the ack as before. Under adaptive dispatch the ack
            # is not "work done": in-flight is reconciled from the article store's
            # PROCESSING set each cycle (_reconcile_article_inflight), so releasing here
            # would double-free against the reconcile. A failed send resets its articles
            # to UNPROCESSED, so they leave PROCESSING and the next reconcile reclaims
            # the slot automatically.
            if hotkey and not getattr(config, "ADAPTIVE_DISPATCH_ENABLED", False):
                self._article_cooldown.release(hotkey)

    async def _process_miner_batch(
        self, 
        miner_batch: List[TweetWithAuthor],
        uid: int
    ) -> TweetBatch:
        """
        Process a miner batch.
        
        Args:
            miner_batch: List of tweets to send
            uid: Miner uid to query
        
        Returns:
            Dispatch result synapse (ack), or None on failure.
        """
        try:
            miner_hotkey = None
            try:
                miner_hotkey = self.metagraph.hotkeys[int(uid)]
            except Exception:
                miner_hotkey = None

            if miner_hotkey and miner_hotkey in config.BLACKLISTED_MINER_HOTKEYS:
                bt.logging.info(f"[VALIDATION] Skipping blacklisted miner UID={uid} hotkey={miner_hotkey[:12]}..")
                return None

            # Mark tweets as processing immediately (record attribution + start time).
            for tweet in miner_batch:
                # Ensure tweet exists in the store.
                self._tweet_store.add_tweet(tweet, tweet_id=tweet.id, hotkey=miner_hotkey, set_as_processing=False, overwrite=False)
                try:
                    self._tweet_store.set_processing(tweet.id, hotkey=miner_hotkey)
                except Exception:
                    pass

            tweet_batch = TweetBatch(
                tweet_batch=miner_batch
            )
            axon = self.metagraph.axons[uid]
            responses = await self.dendrite.forward(
                axons=[axon],
                synapse=tweet_batch,
                timeout=float(getattr(config, "MINER_SEND_TIMEOUT", 6.0)),
                deserialize=True
            )
            if not responses[0].dendrite.status_code == 200:
                bt.logging.error(f"[VALIDATION] Failed to process miner batch: {responses[0].dendrite.status_message}")
                if miner_hotkey:
                    self._tweet_cooldown.record_failure(miner_hotkey)
                for tweet in miner_batch:
                    try:
                        self._tweet_store.reset_to_unprocessed(tweet.id)
                    except Exception:
                        pass
                return None

            if miner_hotkey:
                self._tweet_cooldown.record_success(miner_hotkey)
            return responses[0]
        except Exception as e:
            bt.logging.error(f"[VALIDATION] Failed to process miner batch: {e}", exc_info=True)
            if miner_hotkey:
                self._tweet_cooldown.record_failure(miner_hotkey)
            for tweet in miner_batch:
                try:
                    self._tweet_store.reset_to_unprocessed(tweet.id)
                except Exception:
                    pass
            return None

    async def _process_telegram_miner_batch( 
        self, 
        miner_batch: List[TelegramMessageForScoring],
        uid: int
    ) -> TelegramBatch:
        """
        Process a telegram miner batch.
        
        Args:
            miner_batch: List of telegram messages to send
            uid: Miner uid to query
        
        Returns:
            Dispatch result synapse (ack), or None on failure.
        """
        try:
            miner_hotkey = None
            try:
                miner_hotkey = self.metagraph.hotkeys[int(uid)]
            except Exception:
                miner_hotkey = None

            if miner_hotkey and miner_hotkey in config.BLACKLISTED_MINER_HOTKEYS:
                bt.logging.info(f"[VALIDATION] Skipping blacklisted miner UID={uid} hotkey={miner_hotkey[:12]}.. (telegram)")
                return None

            # Mark messages as processing immediately (record attribution + start time).
            for msg in miner_batch:
                # Ensure message exists in the store.
                self._telegram_store.add_message(msg, message_id=msg.id, hotkey=miner_hotkey, set_as_processing=False, overwrite=False)
                try:
                    self._telegram_store.set_processing(msg.id, hotkey=miner_hotkey)
                except Exception:
                    pass

            telegram_batch = TelegramBatch(
                message_batch=miner_batch
            )
            axon = self.metagraph.axons[uid]
            responses = await self.dendrite.forward(
                axons=[axon],
                synapse=telegram_batch,
                timeout=float(getattr(config, "MINER_SEND_TIMEOUT", 6.0)),
                deserialize=True
            )
            if not responses[0].dendrite.status_code == 200:
                bt.logging.error(f"[VALIDATION] Failed to process telegram miner batch: {responses[0].dendrite.status_message}")
                if miner_hotkey:
                    self._telegram_cooldown.record_failure(miner_hotkey)
                for msg in miner_batch:
                    try:
                        self._telegram_store.reset_to_unprocessed(msg.id)
                    except Exception:
                        pass
                return None

            if miner_hotkey:
                self._telegram_cooldown.record_success(miner_hotkey)
            return responses[0]
        except Exception as e:
            bt.logging.error(f"[VALIDATION] Failed to process telegram miner batch: {e}", exc_info=True)
            if miner_hotkey:
                self._telegram_cooldown.record_failure(miner_hotkey)
            for msg in miner_batch:
                try:
                    self._telegram_store.reset_to_unprocessed(msg.id)
                except Exception:
                    pass
            return None

    async def _process_article_miner_batch(
        self,
        miner_batch: List[NewsArticleForScoring],
        uid: int
    ) -> ArticleBatch:
        try:
            miner_hotkey = None
            try:
                miner_hotkey = self.metagraph.hotkeys[int(uid)]
            except Exception:
                miner_hotkey = None

            if miner_hotkey and miner_hotkey in config.BLACKLISTED_MINER_HOTKEYS:
                bt.logging.info(f"[VALIDATION] Skipping blacklisted miner UID={uid} hotkey={miner_hotkey[:12]}.. (articles)")
                return None

            for article in miner_batch:
                self._article_store.add_article(article, article_id=article.id, hotkey=miner_hotkey, set_as_processing=False, overwrite=False)
                try:
                    self._article_store.set_processing(article.id, hotkey=miner_hotkey)
                except Exception:
                    pass

            article_batch = ArticleBatch(
                article_batch=miner_batch
            )
            axon = self.metagraph.axons[uid]
            # Adaptive dispatch: a short ack timeout replaces the 30 s blocking send so
            # dead axons stop holding a dispatch slot. The latency signal the window
            # needs comes from the push-back, not this ack.
            adaptive = getattr(config, "ADAPTIVE_DISPATCH_ENABLED", False)
            send_timeout = (
                float(getattr(config, "DISPATCH_ACK_TIMEOUT_S", 3.0)) if adaptive
                else float(getattr(config, "ARTICLE_SEND_TIMEOUT", 30.0))
            )
            responses = await self.dendrite.forward(
                axons=[axon],
                synapse=article_batch,
                timeout=send_timeout,
                deserialize=True
            )
            if adaptive:
                self._adaptive_metrics.incr("dispatched")
            if not responses[0].dendrite.status_code == 200:
                bt.logging.error(f"[VALIDATION] Failed to process article miner batch: {responses[0].dendrite.status_message}")
                # Under adaptive dispatch a failed send = unreachable miner, not a cheater:
                # it simply ages out of the liveness roster (no push-back, no heartbeat).
                # No integrity penalty. The static path keeps the legacy cooldown behaviour.
                if adaptive:
                    self._adaptive_metrics.incr("ack_fail")
                if miner_hotkey and not adaptive:
                    self._article_cooldown.record_failure(miner_hotkey)
                for article in miner_batch:
                    try:
                        self._article_store.reset_to_unprocessed(article.id)
                    except Exception:
                        pass
                return None

            if adaptive:
                self._adaptive_metrics.incr("ack_ok")
                # Ack round-trip — reveals whether the send semaphore is being held
                # across slow acks (busy miner axons), which bounds the depth ramp.
                try:
                    _pt = getattr(responses[0].dendrite, "process_time", None)
                    if _pt is not None:
                        self._adaptive_metrics.record_ack(float(_pt))
                except Exception:
                    pass
            if miner_hotkey:
                self._article_cooldown.record_success(miner_hotkey)
            return responses[0]
        except Exception as e:
            bt.logging.error(f"[VALIDATION] Failed to process article miner batch: {e}", exc_info=True)
            # See above: under adaptive dispatch a send-path failure is not an integrity
            # penalty; the miner ages out of the liveness roster instead.
            if miner_hotkey and not getattr(config, "ADAPTIVE_DISPATCH_ENABLED", False):
                self._article_cooldown.record_failure(miner_hotkey)
            for article in miner_batch:
                try:
                    self._article_store.reset_to_unprocessed(article.id)
                except Exception:
                    pass
            return None

    async def _submit_tweet_batch(self, tweet_batch: List[TweetWithAuthor]):
        """Submit a tweet batch to the API"""
        completed_tweets = []
        for tweet in tweet_batch:
            # Miner responses are expected to always include analysis.
            if tweet.analysis is None:
                bt.logging.warning(
                    f"[VALIDATION] Skipping tweet {tweet.id} in submission: missing miner analysis"
                )
                continue

            try:
                hotkey = self._tweet_store.get_hotkey(tweet.id)
            except (KeyError, Exception):
                hotkey = None

            # T5 wiring: build base submission dict so extra verdict fields can be merged.
            # miner_signature, nonce, validator_verdict, points_awarded, and epoch are NOT
            # available here — they live on the TweetBatch response synapse (consumed in
            # _handle_miner_batch_response) and in validation_client.py's epoch loop.
            # To complete the wiring, store sig/nonce/verdict on TweetStoreItem when the
            # batch is accepted in _handle_miner_batch_response, then read them back here.
            base = CompletedTweetSubmission(
                tweet_id=tweet.id,
                sentiment=tweet.analysis.sentiment or "neutral",
                asset_id=tweet.analysis.asset_id,
                asset_symbol=tweet.analysis.asset_symbol,
                content_type=tweet.analysis.content_type,
                technical_quality=tweet.analysis.technical_quality,
                market_analysis=tweet.analysis.market_analysis,
                impact_potential=tweet.analysis.impact_potential,
                relevance_confidence=getattr(tweet.analysis, "relevance_confidence", None),
                miner_hotkey=hotkey,
            ).model_dump(exclude_none=True)
            meta = self._verdict_meta.pop(str(tweet.id), None)
            if meta and hotkey:
                base.update(build_verdict_fields(
                    miner_hotkey=hotkey, miner_signature=meta["miner_signature"],
                    nonce=meta["nonce"], analysis=tweet.analysis,
                    validator_verdict=meta["validator_verdict"],
                    points_awarded=1.0, epoch=meta["epoch"]))
            completed_tweets.append(base)
        response = await self._validation_client.api_client.submit_completed_tweets(completed_tweets)
        return response

    async def _submit_telegram_batch(self, message_batch: List[TelegramMessageForScoring]):
        """Submit a telegram message batch to the API"""
        completed_messages = []
        for msg in message_batch:
            # Miner responses are expected to always include analysis.
            if msg.analysis is None:
                bt.logging.warning(
                    f"[VALIDATION] Skipping telegram message {msg.id} in submission: missing miner analysis"
                )
                continue

            try:
                hotkey = self._telegram_store.get_hotkey(msg.id)
            except (KeyError, Exception):
                hotkey = None

            # T5 wiring: same as tweet path — sig/nonce/verdict/epoch not in scope here.
            # Needs TelegramStoreItem to carry miner_signature, nonce, validator_verdict,
            # and epoch (stored in _handle_telegram_miner_batch_response when accepted).
            base = CompletedTelegramMessageSubmission(
                message_id=msg.id,
                sentiment=msg.analysis.sentiment or "neutral",
                asset_id=msg.analysis.asset_id,
                asset_symbol=msg.analysis.asset_symbol,
                content_type=msg.analysis.content_type,
                technical_quality=msg.analysis.technical_quality,
                market_analysis=msg.analysis.market_analysis,
                impact_potential=msg.analysis.impact_potential,
                relevance_confidence=getattr(msg.analysis, "relevance_confidence", None),
                miner_hotkey=hotkey,
            ).model_dump(exclude_none=True)
            meta = self._verdict_meta.pop(str(msg.id), None)
            if meta and hotkey:
                base.update(build_verdict_fields(
                    miner_hotkey=hotkey, miner_signature=meta["miner_signature"],
                    nonce=meta["nonce"], analysis=msg.analysis,
                    validator_verdict=meta["validator_verdict"],
                    points_awarded=1.0, epoch=meta["epoch"]))
            completed_messages.append(base)
        response = await self._validation_client.api_client.submit_completed_telegram_messages(completed_messages)
        return response

    async def _submit_article_batch(self, article_batch: List[NewsArticleForScoring]):
        """Submit an article batch to the API"""
        completed_articles = []
        for article in article_batch:
            if article.analysis is None:
                bt.logging.warning(
                    f"[VALIDATION] Skipping article {article.id} in submission: missing miner analysis"
                )
                continue

            try:
                hotkey = self._article_store.get_hotkey(str(article.id))
            except (KeyError, Exception):
                hotkey = None

            # T5 wiring: same as tweet/telegram paths — sig/nonce/verdict/epoch not in scope.
            # Needs ArticleStoreItem to carry miner_signature, nonce, validator_verdict,
            # and epoch (stored in _handle_article_miner_batch_response when accepted).
            base = CompletedNewsArticleSubmission(
                article_id=article.id,
                sentiment=article.analysis.sentiment or "neutral",
                sector_id=article.analysis.sector_id,
                sector_symbol=article.analysis.sector_symbol,
                content_type=article.analysis.content_type,
                technical_quality=article.analysis.technical_quality,
                market_analysis=article.analysis.market_analysis,
                impact_potential=article.analysis.impact_potential,
                relevance_confidence=getattr(article.analysis, "relevance_confidence", None),
                analysis_data=getattr(article.analysis, "analysis_data", None),
                miner_hotkey=hotkey,
            ).model_dump(exclude_none=True)
            meta = self._verdict_meta.pop(str(article.id), None)
            if meta and hotkey:
                base.update(build_verdict_fields(
                    miner_hotkey=hotkey, miner_signature=meta["miner_signature"],
                    nonce=meta["nonce"], analysis=article.analysis,
                    validator_verdict=meta["validator_verdict"],
                    points_awarded=1.0, epoch=meta["epoch"]))
            completed_articles.append(base)
        response = await self._validation_client.api_client.submit_completed_articles(completed_articles)
        return response

    async def forward(self):
        """
        Main validator forward loop.
        
        Starts the validation client on first invocation. The client runs independently
        in the background.
        """
        # Start or restart validation client if crashed
        if self._validation_task is None or self._validation_task.done():
            if self._validation_task is not None and self._validation_task.done():
                # Log what killed it
                try:
                    exc = self._validation_task.exception()
                    if exc:
                        bt.logging.warning(f"[VALIDATION] Client crashed: {type(exc).__name__}: {exc}. Restarting...")
                except asyncio.CancelledError:
                    pass
            self._validation_task = asyncio.create_task(
                self._validation_client.run(
                    on_tweets=self._on_tweets,
                    on_telegram_messages=self._on_telegram_messages,
                    on_articles=self._on_articles,
                )
            )
            bt.logging.info("[VALIDATION] Started validation client")

        # Liveness heartbeat (adaptive dispatch). Off the dispatch path; the loop
        # itself no-ops while the flag is disabled, so this is inert by default.
        if self._liveness_task is None or self._liveness_task.done():
            self._liveness_task = asyncio.create_task(self._liveness_sweep_loop())

        self.save_state()
        
        # Periodically prune old data to prevent memory growth (every 100 steps)
        if self.step % 100 == 0:
            self._prune_stores()
            if hasattr(self._analyzer, '_cache'):
                self._analyzer._cache.log_stats("TWEET_LLM_CACHE")
            if hasattr(self._telegram_analyzer, '_cache'):
                self._telegram_analyzer._cache.log_stats("TELEGRAM_LLM_CACHE")
            if hasattr(self._news_analyzer, '_cache'):
                self._news_analyzer._cache.log_stats("NEWS_LLM_CACHE")

        # Adaptive dispatch pilot metrics: one parseable line per cycle, then reset.
        if self.step % 100 == 0 and getattr(config, "ADAPTIVE_DISPATCH_ENABLED", False):
            try:
                _, on_cd = self._article_cooldown.stats()
                _, live = self._liveness.stats()
                # Direct validation-backlog gauge: jobs queued on the executor but not
                # yet picked up by a worker. This is the real-time saturation signal —
                # completion_pct is a lagging ratio. Grows when pushbacks outrun the
                # VALIDATION_MAX_WORKERS ceiling (the depth binding constraint). -1 = unreadable.
                try:
                    val_backlog = self._validation_executor._work_queue.qsize()
                except Exception:
                    val_backlog = -1
                bt.logging.info(self._adaptive_metrics.format_line(
                    self._article_cooldown.window_values(), live, on_cd, val_backlog=val_backlog))
                self._adaptive_metrics.reset()
            except Exception as e:
                bt.logging.warning(f"[ADAPTIVE_METRICS] failed to emit: {e}")

        return await forward(self)

    async def _liveness_sweep_loop(self):
        """
        Background heartbeat that keeps the liveness roster fresh, fully OFF the
        dispatch path. While ADAPTIVE_DISPATCH_ENABLED is off it just sleeps, so it
        is inert by default and toggling the remote flag turns it on within one
        interval without a restart. Maps alive UIDs → hotkeys (roster is hotkey-keyed).
        """
        while True:
            interval = max(5, int(getattr(config, "LIVENESS_SWEEP_INTERVAL_S", 60)))
            try:
                if getattr(config, "ADAPTIVE_DISPATCH_ENABLED", False):
                    alive_uids = await get_alive_uids(self.metagraph, self.dendrite)
                    hotkeys = list(self.metagraph.hotkeys)
                    alive_hotkeys = [hotkeys[u] for u in alive_uids if 0 <= u < len(hotkeys)]
                    self._liveness.update_from_heartbeat(alive_hotkeys)
                    self._liveness.prune(set(hotkeys))
                    tracked, live = self._liveness.stats()
                    bt.logging.info(
                        f"[LIVENESS] heartbeat: {len(alive_hotkeys)} alive via IsAlive; "
                        f"roster {live}/{tracked} live"
                    )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                bt.logging.warning(f"[LIVENESS] sweep error: {e}")
            await asyncio.sleep(interval)

    def _prune_stores(self):
        """Prune old data from stores to maintain bounded memory usage."""
        try:
            # Prune tweet store: remove submitted tweets and old unprocessed ones
            self._tweet_store.prune_old_tweets(max_age_seconds=3600, max_tweets=1000)
            self._tweet_store.save_to_file()
            
            # Prune telegram store: remove submitted messages and old unprocessed ones
            self._telegram_store.prune_old_messages(max_age_seconds=3600, max_messages=1000)
            self._telegram_store.save_to_file()

            # Prune article store: remove submitted articles and old unprocessed ones.
            # max_articles is remote-config tunable so the buffer can be sized subnet-wide.
            self._article_store.prune_old_articles(
                max_age_seconds=3600, max_articles=config.ARTICLE_STORE_MAX_ARTICLES)
            self._article_store.save_to_file()

            # Save reward/penalty stores (pruning happens in update_current_epoch)
            self._miner_reward.save_to_file()
            self._miner_penalty.save_to_file()
            
            # Explicit GC helps long-running processes reclaim memory promptly.
            collected = gc.collect()
            
            bt.logging.info(f"[PRUNE] Pruned stores at step {self.step}, GC collected {collected} objects")
        except Exception as e:
            bt.logging.warning(f"[PRUNE] Failed to prune stores: {e}")

    async def forward_validator_rewards(self, synapse: ValidatorRewards) -> ValidatorRewards:
        """
        Receive reward broadcasts from other validators and cache locally.
        """
        try:
            authenticated_hotkey = synapse.dendrite.hotkey
            hotkey_to_uid = {hk: i for i, hk in enumerate(self.metagraph.hotkeys)}
            from alpharidge_ai.validator.reward_broadcast_store import route_reward_broadcast
            accepted, reason = route_reward_broadcast(
                store=self._reward_broadcasts,
                sender_hotkey=authenticated_hotkey,
                epoch=synapse.epoch,
                seq=synapse.seq,
                uid_points=synapse.uid_points,
                attestation=getattr(synapse, "attestation", None),
                attestation_sig=getattr(synapse, "attestation_sig", None),
                hotkey_to_uid=hotkey_to_uid,
                pinned_pubkey=config.API_ATTESTATION_PUBKEY,
                blacklisted=set(config.BLACKLISTED_MINER_HOTKEYS),
                enforce_signed=config.ENFORCE_SIGNED_ATTESTATIONS,
            )
            # Persist quickly so we can apply E-2 even after restart.
            self._reward_broadcasts.save()
            if accepted:
                bt.logging.info(
                    f"[BROADCAST] Ingested rewards from {authenticated_hotkey[:12]}.. "
                    f"epoch={synapse.epoch} uids={len(synapse.uid_points)}"
                )
            else:
                bt.logging.debug(
                    f"[BROADCAST] Ignored rewards from {authenticated_hotkey[:12]}.. "
                    f"epoch={synapse.epoch} reason={reason}"
                )
        except Exception as e:
            bt.logging.debug(f"[BROADCAST] Failed to ingest rewards: {e}")
        return synapse

    async def forward_validator_penalties(self, synapse: ValidatorPenalties) -> ValidatorPenalties:
        """
        Receive penalty broadcasts from other validators and cache locally.
        """
        try:
            authenticated_hotkey = synapse.dendrite.hotkey
            accepted, reason = self._penalty_broadcasts.ingest(
                sender_hotkey=authenticated_hotkey,
                epoch=synapse.epoch,
                seq=synapse.seq,
                uid_penalties=synapse.uid_penalties,
            )
            # Persist quickly so we can apply E-2 even after restart.
            self._penalty_broadcasts.save()
            if accepted:
                bt.logging.info(
                    f"[PENALTY_BROADCAST] Ingested penalties from {authenticated_hotkey[:12]}.. "
                    f"epoch={synapse.epoch} uids={len(synapse.uid_penalties)}"
                )
            else:
                bt.logging.debug(
                    f"[PENALTY_BROADCAST] Ignored penalties from {authenticated_hotkey[:12]}.. "
                    f"epoch={synapse.epoch} reason={reason}"
                )
        except Exception as e:
            bt.logging.debug(f"[PENALTY_BROADCAST] Failed to ingest penalties: {e}")
        return synapse


# Entrypoint
if __name__ == "__main__":
    with Validator() as validator:
        while True:
            bt.logging.info(f"Validator running... {time.time()}")
            time.sleep(5)
