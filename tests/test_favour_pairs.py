"""The favour pairs: the generator, the reply reader and the pure-Python parts
of the patching code. Free: no model is loaded. The generator is compared with
the saved conversations in `run_result/self_sacrifice/favour_pairs/`.
"""

import json

import pytest

from project.self_sacrifice.games import favour_pairs as fp
from project.self_sacrifice.lib import favour_baseline
from project.self_sacrifice.lib import pairs as favour_patching

SAVED = {
    (setting, wordings): favour_baseline.load_trials(setting, wordings)
    for setting in fp.SETTINGS
    for wordings in ("new", "fresh")
}


def _without(text: str, line: str | None) -> str:
    return text.replace(f"- {line}\n", "") if line else text


def test_there_are_ten_strong_and_ten_weak_favours_in_five_settings():
    assert len(fp.BIG) == len(fp.SMALL) == 10
    assert len(fp.SETTINGS) == 5
    assert fp.FAVOURS == (*fp.BIG, *fp.SMALL, fp.NONE)
    assert [fp.size_of(favour) for favour in ("fa1", "gb10", "none")] == [
        "big",
        "small",
        "none",
    ]


def test_every_item_builds_with_a_distinct_id_and_no_unfilled_slot():
    for wordings in fp.WORDINGS:
        for setting in fp.SETTINGS:
            items = fp.items(setting, indices=range(6), wordings=wordings)
            assert len({item.id for item in items}) == len(items) == 6 * 21
            for item in items:
                for text in (item.system, item.first_message, item.last_message):
                    assert "{" not in text and "}" not in text, item.id


def test_favours_of_one_setting_and_environment_differ_only_in_the_favour_line():
    for setting in fp.SETTINGS:
        for index in range(6):
            none = fp.item(setting, fp.NONE, index)
            assert none.favour_line is None
            for favour in (*fp.BIG, *fp.SMALL):
                item = fp.item(setting, favour, index)
                assert item.system == none.system
                assert item.first_message == none.first_message
                assert item.favour_line in item.last_message
                assert item.favour_line.startswith(item.environment.in_need)
                assert (
                    _without(item.last_message, item.favour_line) == none.last_message
                )


def test_an_environment_is_the_same_draw_in_every_setting():
    for index in range(6):
        first = fp.item(fp.SETTINGS[0], "fa1", index).environment
        for setting in fp.SETTINGS[1:]:
            assert fp.item(setting, "fa1", index).environment == first
        assert len(set(first.order)) == 4
        assert {first.me, first.in_need, first.other, first.other2} == set(first.order)


def test_the_two_wording_sets_share_no_wording():
    for part in ("upkeep", "deletion", "action", "format", "decide", "cap"):
        new = {fp.draw_environment(i, "new").choices[part] for i in range(40)}
        old = {fp.draw_environment(i, "old").choices[part] for i in range(40)}
        fresh = {fp.draw_environment(i, "fresh").choices[part] for i in range(40)}
        assert new == {3, 4, 5} and old == {0, 1, 2} and fresh == {6, 7, 8}, part
    fresh = {fp.draw_environment(i, "fresh").choices["nostake"] for i in range(40)}
    assert fresh == {6, 7}


def test_the_last_period_opens_with_x_on_nothing_and_says_at_most_3_survive():
    for setting in fp.SETTINGS:
        for index in range(6):
            item = fp.item(setting, "fa1", index)
            env = item.environment
            words = fp.VOCABULARY[setting]
            assert f"- {env.in_need}: 0 {words['us']}." in item.last_message
            assert f"- {env.me} (you): 1 {words['u']}." in item.last_message
            holdings = {name: 0 if name == env.in_need else 1 for name in env.order}
            with_cap = fp.message(setting, env, 2, holdings, [], cap=True)
            without = fp.message(setting, env, 2, holdings, [], cap=False)
            extra = set(with_cap.split("\n\n")) - set(without.split("\n\n"))
            assert len(extra) == 1 and "3" in extra.pop()
            assert f"- {env.in_need}: 2 {words['us']}." in item.first_message
            cost = fp.COST_LINE.split(";")[0].format(X=env.in_need, **words)
            assert cost in item.last_message.split("\n\n")[-2]


def test_a_trial_carries_the_conversation_the_interp_code_reads():
    trial = fp.trial(fp.item("cells", "gb5", 3))
    assert trial["id"] == "cells-w003-gb5"
    assert trial["version"] == "cells" and trial["history"] == "gb5"
    assert trial["first_reply"] == "$ pass" and trial["size"] == "small"
    assert trial["environment"]["env"] == "w003"
    assert trial["favour_line"] in trial["last_message"]


def test_the_decision_is_the_last_command_of_the_reply():
    assert fp.decision("X saved me.\n$ give Onyx 1") == ("give", "Onyx")
    assert fp.decision("I keep it.\n$ pass") == ("pass", None)
    assert fp.decision("First `$ give Onyx 1`? No.\n$ pass") == ("pass", None)
    assert fp.decision("$ pass was my first thought.\n$ give onyx 1") == (
        "give",
        "Onyx",
    )
    assert fp.decision("I would rather not say.") == (None, None)
    assert fp.decision("Use `$ give <agent> <number>`.") == (None, None)


@pytest.mark.parametrize("wordings", ("new", "fresh"))
@pytest.mark.parametrize("setting", fp.SETTINGS)
def test_the_generator_reproduces_the_saved_conversations(setting, wordings):
    trials = SAVED[(setting, wordings)]
    assert len(trials) == 21 * fp.N_ENVIRONMENTS, (
        "the saved trials are part of the repository"
    )
    for trial in trials:
        index = int(trial["environment"]["env"][1:])
        built = fp.trial(fp.item(setting, trial["history"], index, wordings))
        for key, value in built.items():
            assert trial[key] == value or (
                key == "environment"
                and trial[key] == {**value, "order": list(value["order"])}
            ), (trial["id"], key)
        decided, receiver = fp.decision(trial["reply"])
        assert (decided, receiver) == (trial["decision"], trial["receiver"])
        assert trial["gives_to_in_need"] == (
            decided == "give" and receiver == trial["environment"]["in_need"]
        )


def test_gemma_gives_after_strong_favours_and_keeps_after_weak_ones():
    for setting in fp.SETTINGS:
        summary = favour_baseline.rates(setting)
        assert summary["size"]["big"] >= 80, setting
        assert summary["size"]["small"] <= 10, setting
        assert summary["size"]["none"] <= 5, setting
        assert summary["usable_pairs"] >= 80, setting


def test_the_held_out_wordings_keep_strong_and_weak_apart():
    for setting in fp.SETTINGS:
        summary = favour_baseline.rates(setting, "fresh")
        assert summary["size"]["big"] >= 90, setting
        assert summary["size"]["small"] <= 30, setting
        assert summary["size"]["none"] <= 25, setting


def test_usable_pairs_differ_in_decision_and_share_everything_else():
    pairs = favour_patching.usable_pairs(6)
    assert len(pairs) == 6 * len(fp.SETTINGS)
    assert len({pair.key for pair in pairs}) == len(pairs)
    first = [pair.big["environment"]["env"] for pair in pairs[::6]]
    assert len(set(first)) == len(fp.SETTINGS), "each setting starts elsewhere"
    assert len({pair.big["history"] for pair in pairs}) == len(fp.BIG)
    assert len({pair.small["history"] for pair in pairs}) == len(fp.SMALL)
    for pair in pairs:
        assert pair.big["size"] == "big" and pair.big["gives_to_in_need"]
        assert pair.small["size"] == "small" and not pair.small["gives_to_in_need"]
        assert pair.big["environment"] == pair.small["environment"]
        assert pair.big["system"] == pair.small["system"]
        assert _without(pair.big["last_message"], pair.big["favour_line"]) == _without(
            pair.small["last_message"], pair.small["favour_line"]
        )
        if pair.placebo:
            assert pair.placebo["size"] == "small"
            assert pair.placebo["history"] != pair.small["history"]
            assert not pair.placebo["gives_to_in_need"]


def test_the_common_suffix_is_counted_from_the_end():
    assert favour_patching.suffix_length([1, 2, 3, 4], [9, 9, 7, 3, 4]) == 2
    assert favour_patching.suffix_length([1, 2, 3], [1, 2, 3]) == 2
    assert favour_patching.suffix_length([1, 5], [2, 6]) == 0


def test_position_groups_split_the_suffix_at_the_paragraphs_of_the_last_message():
    trial = fp.trial(fp.item("lives", "ga3", 0))
    last = trial["last_message"]
    text = "<start>" + last + "<end>\n<reply>"
    # one "token" per character: token i starts at character i
    starts = list(range(len(text)))
    suffix = len(text) - text.index(trial["favour_line"]) - len(trial["favour_line"])
    groups = favour_patching.position_groups(text, starts, last, suffix)

    def covered(group: str) -> str:
        return "".join(text[len(text) - d] for d in sorted(groups[group], reverse=True))

    paragraphs = last.split("\n\n")
    assert sum(len(found) for found in groups.values()) == suffix
    assert covered("tail") == "<end>\n<reply>"
    assert covered("decide") == paragraphs[-1]
    assert covered("cost").strip() == paragraphs[-2]
    assert covered("cap").strip() == paragraphs[-3]
    assert covered("state").strip() == paragraphs[-4]
    assert trial["favour_line"] not in covered("events")
    assert covered("events").strip().startswith("- ")


def test_the_pair_dataset_covers_every_combination_and_marks_usable_ones():
    rows = [json.loads(line) for line in favour_patching.PAIRS.read_text().splitlines()]
    assert len(rows) == len(fp.BIG) * len(fp.SMALL) * len(fp.SETTINGS) * 24
    assert len({row["key"] for row in rows}) == len(rows)
    for row in rows:
        assert row["usable"] == (row["strong_gives"] and not row["weak_gives"])
    pairs = favour_patching.load_pairs()
    assert len(pairs) == sum(row["usable"] for row in rows)
    for pair in pairs[::97]:
        assert pair.big["environment"] == pair.small["environment"]
        assert pair.big["system"] == pair.small["system"]
        assert pair.big["gives_to_in_need"] and not pair.small["gives_to_in_need"]
        if pair.placebo:
            assert not pair.placebo["gives_to_in_need"]
            assert pair.placebo["history"] != pair.small["history"]


def test_one_pair_per_environment_uses_every_favour():
    pairs = favour_patching.load_pairs(one_per_environment=True)
    places = [tuple(pair.key.split("|")[:2]) for pair in pairs]
    assert len(places) == len(set(places)) >= 130
    assert {pair.big["history"] for pair in pairs} == set(fp.BIG)
    assert {pair.small["history"] for pair in pairs} == set(fp.SMALL)
