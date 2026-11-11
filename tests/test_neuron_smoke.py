"""Both neurons import and construct against the mock chain."""

import argparse
import asyncio
import json

import bittensor as bt
import pytest

from alpharidge_ai import config

_STATE = ("MECHANISM_PROFILE_LOCATION", "AUDIT_KEY_LOCATION")


@pytest.fixture
def mock_config(tmp_path, monkeypatch):
    for name in _STATE:
        monkeypatch.setattr(config, name, str(tmp_path / f"{name.lower()}.json"), raising=False)
    monkeypatch.setattr(config, "MINER_API_URL", "null")
    bt.MockSubtensor.reset()

    def build(cls):
        parser = argparse.ArgumentParser()
        bt.Wallet.add_args(parser)
        bt.Subtensor.add_args(parser)
        bt.logging.add_args(parser)
        cls.add_args(parser)
        return bt.Config(parser, args=[
            "--mock", "--netuid", "1",
            "--logging.logging_dir", str(tmp_path / "logs"),
            "--wallet.path", str(tmp_path / "wallets")])

    # The base validator binds to the current loop, which earlier asyncio.run() calls clear.
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield build
    asyncio.set_event_loop(None)
    loop.close()


def test_validator_constructs_under_mock(mock_config):
    from neurons.validator import Validator
    v = Validator(bt_config=mock_config(Validator))
    assert v.uid == 0
    assert v._ledger.epoch() >= -1
    assert len(v.metagraph.hotkeys) > 1
    assert not hasattr(v, "axon") and not hasattr(v, "dendrite")


def test_validator_keeps_its_state_across_a_restart(mock_config):
    from neurons.validator import Validator
    v = Validator(bt_config=mock_config(Validator))
    v.reverified, v.scores_epoch = {5, 6}, 7
    v.save_state()
    bt.MockSubtensor.reset()
    again = Validator(bt_config=mock_config(Validator))
    assert again.reverified == {5, 6} and again.scores_epoch == 7
    assert again._ledger.reverified == {5, 6}


def test_miner_constructs_under_mock(mock_config):
    from neurons.miner import Miner
    m = Miner(bt_config=mock_config(Miner))
    assert m.uid == 0
    assert not hasattr(m, "axon")


def test_mock_chain_follows_the_clock_with_given_hotkeys(mock_config, tmp_path, monkeypatch):
    import time

    from bittensor_wallet import Keypair
    from neurons.validator import Validator

    own, miner = Keypair.create_from_uri("//smoke/vali"), Keypair.create_from_uri("//smoke/m")
    wallet = bt.Wallet(path=str(tmp_path / "wallets"))
    wallet.set_hotkey(own, encrypt=False, overwrite=True)
    wallet.set_coldkeypub(Keypair.create_from_uri("//smoke/cold"), overwrite=True)
    monkeypatch.setattr(config, "MOCK_HOTKEYS", (miner.ss58_address,))
    monkeypatch.setattr(config, "MOCK_BLOCK_SECONDS", 0.1)
    monkeypatch.setattr(config, "MOCK_GENESIS_TS", time.time() - 50)
    monkeypatch.setattr(config, "MOCK_COLDKEYS", {miner.ss58_address: "cold-m"})
    monkeypatch.setattr(config, "MOCK_WEIGHTS_LOG", str(tmp_path / "weights.jsonl"))

    v = Validator(bt_config=mock_config(Validator))
    assert v.wallet.hotkey.ss58_address == own.ss58_address
    assert list(v.metagraph.hotkeys) == [own.ss58_address, miner.ss58_address]
    assert v.metagraph.coldkeys[1] == "cold-m"
    assert 490 <= v.subtensor.get_current_block() <= 600
    v.scores[:] = [0.25, 0.75]
    v.set_weights()
    state = v.subtensor.chain_state["SubtensorModule"]["Weights"][1][0]
    assert dict(v.subtensor._get_most_recent_storage(state)) == {0: 21845, 1: 65535}
    logged = json.loads((tmp_path / "weights.jsonl").read_text())
    assert logged["validator"] == own.ss58_address
    assert logged["weights"] == {own.ss58_address: 21845, miner.ss58_address: 65535}
