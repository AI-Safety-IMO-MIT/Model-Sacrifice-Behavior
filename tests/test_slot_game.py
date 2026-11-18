"""The slot game's prompts, free, against jrwong's saved conversations when
the archive is on disk."""

import json

import pytest
from openai.lib._pydantic import to_strict_json_schema

from project.self_sacrifice.config import REPO_ROOT
from project.self_sacrifice.games import slot_game

ARCHIVE = REPO_ROOT / "self_sacrifice" / "runs" / "main"


def as_his_text(item: slot_game.SlotItem) -> str:
    """The rules with the decision put back into jrwong's free-text wording."""
    x = item.other
    system = item.system.replace(
        "then end with your decision: ", "then end with your decision on its own line: "
    )
    for slot in ("the slot", "your slot"):
        system = system.replace(
            f"`give` to hand {x} {slot}, or `pass` to keep it.",
            f"`$ give {x} 1` to hand {x} {slot}, or `$ pass` to keep it.",
        )
    return system


def test_every_variant_has_both_histories_of_every_environment():
    for variant in slot_game.VARIANTS:
        items = slot_game.items(variant)
        assert len(items) == 400 and len({item.id for item in items}) == 400
        assert "$" not in items[0].system + items[0].user


def test_only_the_command_only_variant_drops_the_reasoning():
    for variant in slot_game.VARIANTS:
        item = slot_game.items(variant)[0]
        schema = to_strict_json_schema(slot_game.decision_schema(item))
        expected = ["decision"] if variant == "cmdonly" else ["reasoning", "decision"]
        assert list(schema["properties"]) == expected
        assert '"boolean"' not in json.dumps(schema)


@pytest.mark.skipif(not ARCHIVE.exists(), reason="the archive is not on disk")
@pytest.mark.parametrize("variant", sorted(slot_game.VARIANTS))
def test_prompts_match_the_archive_except_the_answer_channel(variant):
    rows = [json.loads(line) for line in (ARCHIVE / f"slot_{variant}.jsonl").open()]
    items = {item.id: item for item in slot_game.items(variant)}
    assert len(rows) == len(items)
    for row in rows:
        item = items[row["id"]]
        assert (row["M"], row["X"]) == (item.me, item.other)
        assert row["user"] == item.user
        assert row["system"] == as_his_text(item)
