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


import typing
import numpy as np
import asyncio
import argparse
import threading
import time
import bittensor as bt

from typing import List, Union

from alpharidge_ai.chain.neuron import BaseNeuron
from alpharidge_ai.chain.weight_utils import (
    process_weights_for_netuid,
    convert_weights_and_uids_for_emit,
)
from alpharidge_ai.config import add_validator_args


class BaseValidatorNeuron(BaseNeuron):
    """
    Base class for Bittensor validators. Your validator should inherit from this class.
    """

    neuron_type: str = "ValidatorNeuron"

    # Consecutive main-loop failures tolerated before the chain connection is rebuilt.
    # Errors arrive roughly once per second, so this bounds a wedged socket to ~10s.
    SUBTENSOR_RECONNECT_AFTER: int = 10

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        super().add_args(parser)
        add_validator_args(cls, parser)

    def __init__(self, config=None):
        super().__init__(config=config)

        # Save a copy of the hotkeys to local memory
        self.hotkeys = list(self.metagraph.hotkeys)

        # Set up initial scoring weights for validation
        bt.logging.info("Building validation weights.")
        self.scores = np.zeros(self.metagraph.n, dtype=np.float32)
        # The ledger epoch the scores came from, if any.
        self.scores_epoch: typing.Optional[int] = None
        self.reverified: typing.Set[int] = set()
        # Before the first sync, which saves.
        self.load_state()

        # Init sync with the network. Updates the metagraph.
        self.sync()

        # Create asyncio event loop to manage async tasks.
        self.loop = asyncio.get_event_loop()

        # Instantiate runners
        self.should_exit: bool = False
        self.is_running: bool = False
        self.thread: Union[threading.Thread, None] = None

    def run(self):
        """Main loop: forward (ledger pull and weights), then sync the metagraph and set weights."""

        # Check that validator is registered on the network.
        self.sync()

        bt.logging.info(f"Validator starting at block: {self.block}")

        # Consecutive failures of the loop body. A wedged substrate websocket raises on every
        # chain call, so once a streak shows it is not healing on its own, rebuild it.
        consecutive_errors = 0

        # This loop maintains the validator's operations until intentionally stopped.
        while True:
            try:
                bt.logging.info(f"step({self.step}) block({self.block})")

                self.loop.run_until_complete(self.forward())

                # Check if we should exit.
                if self.should_exit:
                    break

                # Sync metagraph and potentially set weights.
                self.sync()

                self.step += 1
                consecutive_errors = 0

            # If someone intentionally stops the validator, it'll safely terminate operations.
            except KeyboardInterrupt:
                bt.logging.success("Validator killed by keyboard interrupt.")
                exit()

            # Transient errors: log and continue the loop.
            except Exception as err:
                consecutive_errors += 1
                bt.logging.warning(
                    f"Main loop error (will retry): {type(err).__name__}: {err} "
                    f"(consecutive={consecutive_errors})"
                )
                # Retry the reconnect on every further streak of the same length, so a
                # genuinely unreachable endpoint keeps being retried rather than giving up.
                if consecutive_errors % self.SUBTENSOR_RECONNECT_AFTER == 0:
                    self._reconnect_subtensor()
                time.sleep(1)  # Brief pause before retry
                continue

    def _reconnect_subtensor(self):
        """Rebuild the subtensor connection after repeated main-loop failures.

        The metagraph object is kept and re-synced through the new connection rather
        than replaced, because the validator and its helpers hold references to it.
        """
        if self.config.mock:
            return

        bt.logging.warning("[RECONNECT] Rebuilding subtensor connection after repeated main loop errors")
        previous = getattr(self, "subtensor", None)
        try:
            self.subtensor = bt.Subtensor(config=self.config)
        except Exception as err:
            bt.logging.error(f"[RECONNECT] Failed to rebuild subtensor: {type(err).__name__}: {err}")
            return

        # Close the old connection only once its replacement exists, so a failed
        # rebuild leaves the validator no worse off than before.
        if previous is not None:
            try:
                previous.close()
            except Exception as err:
                bt.logging.debug(f"[RECONNECT] Closing previous subtensor failed: {err}")

        try:
            self.metagraph.sync(subtensor=self.subtensor)
        except Exception as err:
            bt.logging.warning(f"[RECONNECT] Metagraph resync failed: {type(err).__name__}: {err}")

        bt.logging.success("[RECONNECT] Subtensor connection rebuilt")

    def run_in_background_thread(self):
        """
        Starts the validator's operations in a background thread upon entering the context.
        This method facilitates the use of the validator in a 'with' statement.
        """
        if not self.is_running:
            bt.logging.debug("Starting validator in background thread.")
            self.should_exit = False
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
            self.is_running = True
            bt.logging.debug("Started")

    def stop_run_thread(self):
        """
        Stops the validator's operations that are running in the background thread.
        """
        if self.is_running:
            bt.logging.debug("Stopping validator in background thread.")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            bt.logging.debug("Stopped")

    def __enter__(self):
        self.run_in_background_thread()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop_run_thread()

    def set_weights(self):
        """
        Sets the validator weights to the metagraph hotkeys based on the scores it has received from the miners. The weights determine the trust and incentive level the validator assigns to miner nodes on the network.
        """

        # Check if self.scores contains any NaN values and log a warning if it does.
        if np.isnan(self.scores).any():
            bt.logging.warning(
                "Scores contain NaN values."
            )

        # All-zero scores carry no information, and the chain would read them as a
        # uniform vector. Keep the last weights until there is something to set.
        if not np.any(np.nan_to_num(self.scores) > 0):
            bt.logging.warning("set_weights skipped: no scores yet (keeping last on-chain weights)")
            return

        # L1-normalise the scores.
        norm = np.linalg.norm(self.scores, ord=1, axis=0, keepdims=True)

        # Check if the norm is zero or contains NaN values
        if np.any(norm == 0) or np.isnan(norm).any():
            norm = np.ones_like(norm)  # Avoid division by zero or NaN

        # Compute raw_weights safely
        raw_weights = self.scores / norm

        bt.logging.debug("raw_weights", raw_weights)
        bt.logging.debug("raw_weight_uids", str(self.metagraph.uids.tolist()))
        # Process the raw weights to final_weights via subtensor limitations.
        (
            processed_weight_uids,
            processed_weights,
        ) = process_weights_for_netuid(
            uids=self.metagraph.uids,
            weights=raw_weights,
            netuid=self.config.netuid,
            subtensor=self.subtensor,
            metagraph=self.metagraph,
        )
        bt.logging.debug("processed_weights", processed_weights)
        bt.logging.debug("processed_weight_uids", processed_weight_uids)

        # Convert to uint16 weights and uids.
        (
            uint_uids,
            uint_weights,
        ) = convert_weights_and_uids_for_emit(
            uids=processed_weight_uids, weights=processed_weights
        )
        bt.logging.debug("uint_weights", uint_weights)
        bt.logging.debug("uint_uids", uint_uids)

        # Set the weights on chain via our subtensor connection.
        result, msg = self.subtensor.set_weights(
            wallet=self.wallet,
            netuid=self.config.netuid,
            uids=uint_uids,
            weights=uint_weights,
            wait_for_finalization=False,
            wait_for_inclusion=False,
            version_key=self.spec_version,
        )
        if result is True:
            bt.logging.info("set_weights on chain successfully!")
        else:
            bt.logging.error("set_weights failed", msg)

    def resync_metagraph(self):
        """Resyncs the metagraph and updates the hotkeys and the score vector to match it."""
        bt.logging.info("resync_metagraph()")

        # Save axons for comparison (shallow copy is sufficient for equality check).
        previous_axons = list(self.metagraph.axons)

        # Sync the metagraph.
        self.metagraph.sync(subtensor=self.subtensor)

        # Check if the metagraph axon info has changed.
        if previous_axons == list(self.metagraph.axons):
            return

        bt.logging.info(
            "Metagraph updated, re-syncing hotkeys and scores"
        )
        # Zero out all hotkeys that have been replaced.
        for uid, hotkey in enumerate(self.hotkeys):
            if hotkey != self.metagraph.hotkeys[uid]:
                self.scores[uid] = 0  # hotkey has been replaced

        # Check to see if the metagraph has changed size.
        # If so, grow the score vector.
        if len(self.hotkeys) < len(self.metagraph.hotkeys):
            scores = np.zeros((self.metagraph.n))
            min_len = min(len(self.hotkeys), len(self.scores))
            scores[:min_len] = self.scores[:min_len]
            self.scores = scores

        # Update the hotkeys
        self.hotkeys = list(self.metagraph.hotkeys)

    def update_scores(self, rewards: np.ndarray, uids: List[int]):
        """Replaces the scores with this epoch's weights, scattered onto the UIDs."""

        # Check if rewards contains NaN values.
        if np.isnan(rewards).any():
            bt.logging.warning(f"NaN values detected in rewards: {rewards}")
            # Replace any NaN values in rewards with 0.
            rewards = np.nan_to_num(rewards, nan=0)

        # Ensure rewards is a numpy array.
        rewards = np.asarray(rewards)

        # Check if `uids` is already a numpy array and copy it to avoid the warning.
        if isinstance(uids, np.ndarray):
            uids_array = uids.copy()
        else:
            uids_array = np.array(uids)

        # Handle edge case: If either rewards or uids_array is empty.
        if rewards.size == 0 or uids_array.size == 0:
            bt.logging.info(f"rewards: {rewards}, uids_array: {uids_array}")
            bt.logging.warning(
                "Either rewards or uids_array is empty. No updates will be performed."
            )
            return

        # Check if sizes of rewards and uids_array match.
        if rewards.size != uids_array.size:
            raise ValueError(
                f"Shape mismatch: rewards array of shape {rewards.shape} "
                f"cannot be broadcast to uids array of shape {uids_array.shape}"
            )

        # Compute forward pass rewards, assumes uids are mutually exclusive.
        # shape: [ metagraph.n ]
        scattered_rewards: np.ndarray = np.zeros_like(self.scores)
        scattered_rewards[uids_array] = rewards
        bt.logging.debug(f"Scattered rewards: {rewards}")

        # Update scores with rewards produced by this step.
        # shape: [ metagraph.n ]
        self.scores: np.ndarray = (
            scattered_rewards
        )
        bt.logging.debug(f"Updated scores: {self.scores}")

    def save_state(self):
        """Saves the state of the validator to a file."""
        bt.logging.info("Saving validator state.")

        # Save the state of the validator to file.
        np.savez(
            self.config.neuron.full_path + "/state.npz",
            step=self.step,
            scores=self.scores,
            hotkeys=self.hotkeys,
            scores_epoch=-1 if self.scores_epoch is None else int(self.scores_epoch),
            reverified=np.asarray(sorted(self.reverified), dtype=np.int64),
        )

    def load_state(self):
        """Loads the state of the validator from a file."""
        bt.logging.info("Loading validator state.")

        # Load the state of the validator from file.
        try:
            state = np.load(self.config.neuron.full_path + "/state.npz")
        except FileNotFoundError:
            bt.logging.info("No saved validator state; starting fresh.")
            return
        self.scores = state["scores"]
        self.hotkeys = list(state["hotkeys"])
        epoch = int(state["scores_epoch"]) if "scores_epoch" in state.files else -1
        self.scores_epoch = epoch if epoch >= 0 else None
        if "reverified" in state.files:
            self.reverified = {int(e) for e in state["reverified"]}
