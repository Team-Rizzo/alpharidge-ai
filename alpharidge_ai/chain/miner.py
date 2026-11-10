# The MIT License (MIT)
# Copyright © 2023 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import asyncio
import argparse
import threading
import time
import traceback
from typing import Union

import bittensor as bt

from alpharidge_ai.chain.neuron import BaseNeuron
from alpharidge_ai.config import add_miner_args


class BaseMinerNeuron(BaseNeuron):
    """Base class for miners. Miners push claims to the API, so there is no axon."""

    neuron_type: str = "MinerNeuron"
    _last_sync: int = 0

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        super().add_args(parser)
        add_miner_args(cls, parser)

    def __init__(self, config=None):
        super().__init__(config=config)
        self.should_exit: bool = False
        self.is_running: bool = False
        self.thread: Union[threading.Thread, None] = None
        self.loop = asyncio.new_event_loop()

    async def forward(self):
        """One pass of the miner's work."""
        await asyncio.sleep(1)

    def run(self):
        """Run `forward` until stopped, resyncing the metagraph every epoch."""
        self.sync()
        bt.logging.info(f"Miner starting at block: {self.block}")
        asyncio.set_event_loop(self.loop)
        while not self.should_exit:
            try:
                self.loop.run_until_complete(self.forward())
                if self.should_sync_metagraph():
                    self.sync()
                self.step += 1
            except KeyboardInterrupt:
                bt.logging.success("Miner killed by keyboard interrupt.")
                break
            except Exception:
                bt.logging.error(traceback.format_exc())
                time.sleep(5)

    def run_in_background_thread(self):
        if not self.is_running:
            self.should_exit = False
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
            self.is_running = True

    def stop_run_thread(self):
        if self.is_running:
            self.should_exit = True
            if self.thread is not None:
                self.thread.join(5)
            self.is_running = False

    def __enter__(self):
        self.run_in_background_thread()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop_run_thread()

    def should_sync_metagraph(self):
        return self.block - self._last_sync > self.config.neuron.epoch_length

    def resync_metagraph(self):
        bt.logging.info("resync_metagraph()")
        self.metagraph.sync(subtensor=self.subtensor)
        self._last_sync = self.block
