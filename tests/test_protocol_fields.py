from alpharidge_ai.protocol import TweetBatch, TelegramBatch, ArticleBatch, ValidatorRewards


def test_batches_carry_miner_signatures_and_nonces():
    tb = TweetBatch(tweet_batch=[], miner_signatures={"1": "ab"}, nonces={"1": "n"})
    assert tb.miner_signatures == {"1": "ab"} and tb.nonces == {"1": "n"}
    assert TelegramBatch(message_batch=[]).miner_signatures == {}
    assert ArticleBatch(article_batch=[]).nonces == {}


def test_validator_rewards_carries_optional_attestation():
    vr = ValidatorRewards(epoch=7, uid_points={1: 5}, sender_hotkey="hk", seq=7,
                          attestation={"validatorHotkey": "hk", "epoch": 7},
                          attestation_sig="deadbeef")
    assert vr.attestation["epoch"] == 7
    assert vr.attestation_sig == "deadbeef"
    assert ValidatorRewards(epoch=7, uid_points={}, sender_hotkey="hk", seq=7).attestation is None
