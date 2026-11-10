"""Reference miner entrypoint."""

# Recent bittensor releases ignore CLI args (--netuid/--wallet) unless this is set before import.
import os
os.environ.setdefault("BT_NO_PARSE_CLI_ARGS", "0")

from alpharidge_ai.packages import repair_substrate_codec
repair_substrate_codec()

import asyncio
import time

import bittensor as bt

from alpharidge_ai import config
from alpharidge_ai.chain.miner import BaseMinerNeuron
from alpharidge_ai.ledger.api_client import AlpharidgeAPIClient
from alpharidge_ai.miner.build import ClaimBuilder
from alpharidge_ai.miner.discover import Discoverer, load_sources
from alpharidge_ai.miner.link import Linker
from alpharidge_ai.miner.loop import MinerLoop
from alpharidge_ai.miner.store import MinerStore
from alpharidge_ai.verify.fetch import FetchConfig, Fetcher, user_agent


class Miner(BaseMinerNeuron):
    """SN45 reference miner: discovers events in public feeds and pushes signed claims."""

    def __init__(self, bt_config=None):
        # NOTE: this arg name must not shadow the imported `alpharidge_ai.config` module.
        super().__init__(config=bt_config)
        self._api = AlpharidgeAPIClient(base_url=config.MINER_API_URL, wallet=self.wallet,
                                        timeout=config.BATCH_HTTP_TIMEOUT)
        self._miner = build_loop(self._api, self.config.neuron.full_path)
        bt.logging.info(f"[Miner] started with hotkey: {self.wallet.hotkey.ss58_address}")

    async def forward(self):
        sent = await self._miner.tick()
        bt.logging.info(f"[Miner] pass done, {sent} claims sent")
        await asyncio.sleep(config.MINER_POLL_SECONDS)


def build_loop(client, state_dir: str) -> MinerLoop:
    fetcher = Fetcher(FetchConfig(user_agent=config.MINER_USER_AGENT
                                      or user_agent(config.USER_AGENT_CONTACT),
                                      timeout_s=config.MINER_FETCH_TIMEOUT_S,
                                      domain_rps=config.MINER_DOMAIN_RPS,
                                      public_only=config.MINER_FETCH_PUBLIC_ONLY))
    discoverer = Discoverer(load_sources(config.MINER_FEEDS or None), fetcher,
                            max_age_h=config.MINER_MAX_AGE_H, categories=config.MINER_CATEGORIES)
    linker = Linker()
    builder = ClaimBuilder(linker, categories=config.MINER_CATEGORIES,
                           lead_sentences=config.MINER_LEAD_SENTENCES)
    store = MinerStore(config.MINER_STATE_PATH or os.path.join(state_dir, "miner.sqlite"))
    return MinerLoop(client, store, discoverer, builder, fetcher,
                     items_per_tick=config.MINER_ITEMS_PER_TICK,
                     status_per_tick=config.MINER_STATUS_PER_TICK)


if __name__ == "__main__":
    with Miner() as miner:
        while True:
            bt.logging.info(f"Miner running... {time.time()}")
            time.sleep(60)
