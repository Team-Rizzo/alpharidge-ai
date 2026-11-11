"""Validator entrypoint."""

# Recent bittensor releases ignore CLI args (--netuid/--wallet) unless this is set before import.
import os
os.environ.setdefault("BT_NO_PARSE_CLI_ARGS", "0")

from alpharidge_ai.packages import repair_substrate_codec
repair_substrate_codec()

import asyncio
import time

import bittensor as bt
from alpharidge_ai import config
from alpharidge_ai.chain.validator import BaseValidatorNeuron
from alpharidge_ai.judge.client import from_env
from alpharidge_ai.ledger.api_client import AlpharidgeAPIClient
from alpharidge_ai.validator import audit_key
from alpharidge_ai.validator.judge import JudgeWorker
from alpharidge_ai.validator.loop import LedgerLoop
from alpharidge_ai.validator.profile_client import ProfileClient
from alpharidge_ai.validator.reverify import Reverifier
from alpharidge_ai.validator.scan import Scanner
from alpharidge_ai.validator.selector import Selector
from alpharidge_ai.validator.watchdog import stall


class Validator(BaseValidatorNeuron):
    """Validator neuron for SN45."""

    def __init__(self, bt_config=None):
        # NOTE: this arg name must not shadow the imported `alpharidge_ai.config` module.
        super(Validator, self).__init__(config=bt_config)
        config.check_fetch_guard(bool(self.config.mock), getattr(
            getattr(self.config, "subtensor", None), "network", None))

        self._api = AlpharidgeAPIClient(base_url=config.MINER_API_URL, wallet=self.wallet,
                                        timeout=config.BATCH_HTTP_TIMEOUT)

        config.set_wallet(self.wallet)
        config.refresh_remote_config(force=True)

        # Mechanism profile: capacity and window resolve from here, by block height.
        self._mechanism_profile = ProfileClient()
        self._mechanism_profile.load()

        reverifier = Reverifier(self._api, Selector(audit_key.load()))
        self._scan = Scanner(self._api, fetcher=reverifier.fetcher)
        self._judge_setup = self._judge_settings()
        judge = from_env({**os.environ, **self._judge_setup})
        if judge is None:
            bt.logging.info("[JUDGE] no page judge set up; this validator judges no pages")
        self._judge = JudgeWorker(self._api, judge, fetcher=reverifier.fetcher)
        self._last_step = time.monotonic()
        self._ledger = LedgerLoop(
            self._api, self.metagraph, lambda: int(self.block),
            reverifier=reverifier,
            profile=self._mechanism_profile,
            accepted_epoch=self.scores_epoch, reverified=self.reverified,
            owner=None if self.config.mock else self._subnet_owner)

    @staticmethod
    def _judge_settings() -> dict:
        return {"EL_JUDGE_URL": config.JUDGE_URL, "EL_JUDGE_MODEL": config.JUDGE_MODEL,
                "EL_JUDGE_API_KEY": config.JUDGE_API_KEY}

    def _refresh_judge(self) -> None:
        """Rebuild the page judge when the served endpoint or model changes."""
        setup = self._judge_settings()
        if setup == self._judge_setup:
            return
        self._judge_setup = setup
        try:
            self._judge.judge = from_env({**os.environ, **setup})
        except RuntimeError as e:
            bt.logging.warning(f"[JUDGE] served judge settings rejected: {e}")
            self._judge.judge = None
        bt.logging.info("[JUDGE] page judge " + ("updated" if self._judge.judge else "off"))

    def _subnet_owner(self):
        return self.subtensor.get_subnet_owner_hotkey(self.config.netuid)

    def set_weights(self):
        ledger = getattr(self, "_ledger", None)
        if ledger is None or ledger.held:
            bt.logging.warning("[LEDGER] set_weights skipped: burn UID owner check not passed")
            return
        super().set_weights()

    def stalled(self):
        """Why the validator has stopped working, or None."""
        judging = self._judge.judge is not None
        return stall(time.monotonic(), self._last_step,
                     self._judge.last_pull if judging else None)

    async def forward(self):
        self._last_step = time.monotonic()
        block = int(self.block)
        try:
            await asyncio.to_thread(self._mechanism_profile.refresh, block)
            self._mechanism_profile.resolve(block)
        except Exception as e:
            bt.logging.debug(f"[PROFILE] refresh failed: {e}")
        self._refresh_judge()

        self._scan.tick()
        self._judge.tick()
        weights = await self._ledger.tick()
        self.reverified = self._ledger.reverified
        if weights is not None:
            self.update_scores(weights, self.metagraph.uids.tolist())
            self.scores_epoch = self._ledger.accepted_epoch
        await asyncio.sleep(float(config.VALIDATION_POLL_SECONDS))


# Entrypoint
if __name__ == "__main__":
    with Validator() as validator:
        while True:
            bt.logging.info(f"Validator running... {time.time()}")
            why = validator.stalled()
            if why:
                # pm2 starts the process again; a restart clears a stuck connection or thread.
                bt.logging.error(f"[WATCHDOG] {why}; exiting so the process is restarted")
                os._exit(1)
            time.sleep(5)
