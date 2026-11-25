"""The neurons' env files, and the opt-out for processes that only import parts of the
subnet (the API's tests)."""

from alpharidge_ai import config


def test_env_files_load_unless_skipped(tmp_path, monkeypatch, capsys):
    miner, vali = tmp_path / ".miner_env", tmp_path / ".vali_env"
    miner.write_text("AR_TEST_ENV_A=miner\nAR_TEST_ENV_B=miner\n")
    vali.write_text("AR_TEST_ENV_B=vali\n")
    for k in ("AR_TEST_ENV_A", "AR_TEST_ENV_B"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("ALPHARIDGE_SKIP_ENV_FILES", "1")
    assert config.load_env_files((miner, vali)) == []
    assert "AR_TEST_ENV_A" not in config.os.environ
    # Skipping says so, once, naming the files left unread.
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1 and "ALPHARIDGE_SKIP_ENV_FILES" in out[0] and str(vali) in out[0]
    # Unset, empty or 0 loads them as the neurons do, the validator's file last.
    for off in (None, "", "0"):
        if off is None:
            monkeypatch.delenv("ALPHARIDGE_SKIP_ENV_FILES")
        else:
            monkeypatch.setenv("ALPHARIDGE_SKIP_ENV_FILES", off)
        monkeypatch.setenv("AR_TEST_ENV_A", "shell")
        assert config.load_env_files((miner, vali)) == [miner, vali]
        assert (config.os.environ["AR_TEST_ENV_A"], config.os.environ["AR_TEST_ENV_B"]) == (
            "miner", "vali")


def test_the_neurons_load_the_checkout_env_files_by_default():
    assert config.load_env_files.__defaults__ == ((config._MINER_ENV_PATH,
                                                   config._VALI_ENV_PATH),)
