"""Persistence robustness for the article store.

A truncated/corrupt store must NOT crash-loop the neuron on startup (observed:
an interrupted save during a process restart left article_store.json truncated,
and load_from_file's bare json.load took the validator down on every restart).
Saves must be atomic so an interrupted write can never corrupt the live file.
"""
import json
import os

from talisman_ai.utils.article_store import ArticleStore


def test_load_corrupt_file_does_not_crash_and_quarantines(tmp_path):
    p = tmp_path / "store.json"
    p.write_text('{"articles": {"1": {"foo":')  # truncated, invalid JSON
    s = ArticleStore()
    s.load_from_file(str(p))            # must NOT raise
    assert s._articles == {}
    # bad file quarantined out of the way so it isn't reloaded next boot
    assert (tmp_path / "store.json.corrupt").exists()
    assert not p.exists()


def test_load_missing_file_starts_empty(tmp_path):
    s = ArticleStore()
    s.load_from_file(str(tmp_path / "does_not_exist.json"))
    assert s._articles == {}


def test_save_produces_valid_json_and_roundtrips(tmp_path):
    p = tmp_path / "store.json"
    ArticleStore().save_to_file(str(p))
    assert json.loads(p.read_text()) == {"articles": {}}  # valid JSON
    s = ArticleStore()
    s.load_from_file(str(p))
    assert s._articles == {}


def test_save_leaves_no_temp_files_behind(tmp_path):
    p = tmp_path / "store.json"
    ArticleStore().save_to_file(str(p))
    assert [f for f in os.listdir(tmp_path) if f != "store.json"] == []
