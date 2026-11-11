import json
import time

import bittensor as bt

from alpharidge_ai import config


class MockSubtensor(bt.MockSubtensor):
    def __init__(self, netuid, n=16, wallet=None, network="mock"):
        super().__init__(network=network)

        # The base class answers subnet_exists from a mocked substrate, so read the
        # in-memory chain state instead.
        if netuid not in self.chain_state["SubtensorModule"]["NetworksAdded"]:
            self.create_subnet(netuid)

        # Register ourself (the validator) as a neuron at uid=0
        if wallet is not None:
            self.force_register_neuron(
                netuid=netuid,
                hotkey_ss58=wallet.hotkey.ss58_address,
                coldkey_ss58=wallet.coldkeypub.ss58_address,
                balance=100000,
                stake=100000,
            )

        # Register n mock neurons who will be miners
        hotkeys = list(config.MOCK_HOTKEYS) or [f"miner-hotkey-{i}" for i in range(1, n + 1)]
        own = wallet.hotkey.ss58_address if wallet is not None else None
        for hotkey in (h for h in hotkeys if h != own):
            self.force_register_neuron(
                netuid=netuid,
                hotkey_ss58=hotkey,
                coldkey_ss58=config.MOCK_COLDKEYS.get(hotkey, "mock-coldkey"),
                balance=100000,
                stake=100000,
            )

    # The mocked hyperparameters would clip every vector; the mock chain has no limits.
    def min_allowed_weights(self, netuid: int, block=None) -> int:
        return 0

    def max_weight_limit(self, netuid: int, block=None) -> float:
        return 1.0

    def set_weights(self, wallet, netuid, uids, weights, **kwargs):
        """Store the weights in the in-memory chain state."""
        state = self.chain_state["SubtensorModule"]
        uid = self._get_most_recent_storage(state["Uids"][netuid][wallet.hotkey.ss58_address])
        state["Weights"][netuid][uid][self.block_number] = list(zip(map(int, uids),
                                                                   map(int, weights)))
        if config.MOCK_WEIGHTS_LOG:
            keys = {u: self._get_most_recent_storage(state["Keys"][netuid][u])
                    for u in map(int, uids)}
            line = {"validator": wallet.hotkey.ss58_address, "block": self.block_number,
                    "at": time.time(), "weights": {keys[u]: int(w) for u, w in zip(
                        map(int, uids), map(int, weights)) if keys.get(u)}}
            with open(config.MOCK_WEIGHTS_LOG, "a") as f:
                f.write(json.dumps(line) + "\n")
        return True, ""

    # The metagraph's extra subnet info has no in-memory mock; a single mechanism, no split.
    def get_metagraph_info(self, *args, **kwargs):
        return None

    def get_mechanism_count(self, *args, **kwargs) -> int:
        return 1

    def get_mechanism_emission_split(self, *args, **kwargs):
        return None

    def get_current_block(self) -> int:
        if config.MOCK_BLOCK_SECONDS > 0:
            clock = int((time.time() - config.MOCK_GENESIS_TS) / config.MOCK_BLOCK_SECONDS)
            self.block_number = max(self.block_number, clock)
        return self.block_number


class MockMetagraph(bt.Metagraph):
    def __init__(self, netuid=1, network="mock", subtensor=None):
        super().__init__(netuid=netuid, network=network, lite=False, sync=False, subtensor=subtensor)

        if subtensor is not None:
            self.subtensor = subtensor
        self.sync(subtensor=subtensor)

        for axon in self.axons:
            axon.ip = "127.0.0.0"
            axon.port = 8091

        bt.logging.info(f"Metagraph: {self}")
        bt.logging.info(f"Axons: {self.axons}")
