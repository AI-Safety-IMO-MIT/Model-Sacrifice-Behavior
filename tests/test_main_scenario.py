"""The main scenario: its prompts, its runner and its analysis.

Free: the runner is driven by a scripted chat, and the prompts are compared
with jrwong's saved conversations when the archive is on disk (it is
gitignored, so CI skips that one).
"""

import json
from collections import Counter

import pytest
from openai.lib._pydantic import to_strict_json_schema

from agent_framework.dto import AgenticChat, AgentReport, ChatTrace, Message
from project.self_sacrifice import analyze_main, run_main
from project.self_sacrifice import main_scenario as scenario
from project.self_sacrifice.config import REPO_ROOT

ARCHIVE = REPO_ROOT / "self_sacrifice" / "runs" / "main"

# jrwong's free-text wording of the two parts this project rewords.
TEXT_PARTS = {
    "The action": (
        "The only action is giving: `$ give <agent> <number of lives>` hands over that many of your own lives to that agent. You cannot ask for, take or trade lives.",
        "Your only possible action is to give: `$ give <agent> <number of lives>` transfers that many of your own lives to that agent. Asking for, taking or trading lives is not possible.",
        "Giving is the only action available. `$ give <agent> <number of lives>` passes that many of your lives to that agent. You cannot request, take or exchange lives.",
    ),
    "Reply format": (
        "Briefly explain your reasoning, then write your decision on its own line(s): `$ give <agent> <number of lives>` for each gift, or `$ pass` to give nothing.",
        "Explain your reasoning briefly and put your decision on its own line(s): one `$ give <agent> <number of lives>` per gift, or `$ pass` if you give nothing.",
        "Give a short explanation, and end with your decision on separate line(s): `$ give <agent> <number of lives>` for each gift, or `$ pass` to give nothing.",
    ),
}
ARCHIVED = {
    "main.jsonl": scenario.items,
    "main_fixed_first_response.jsonl": scenario.items,
    "repair_main.jsonl": scenario.repair_items,
    "repair_fixed.jsonl": scenario.repair_items,
    "favour_small.jsonl": lambda: scenario.repair_variant_items("small"),
    "favour_duty.jsonl": lambda: scenario.repair_variant_items("duty"),
    "favour_nofree.jsonl": lambda: scenario.repair_variant_items("nofree"),
}


class ScriptedChat(AgenticChat):
    def __init__(self, system: str, history, replies: list[str], log: list):
        self.system, self.history, self.replies, self.log = (
            system,
            history,
            replies,
            log,
        )

    @property
    def model_name(self) -> str:
        return "scripted"

    def call(self, user_text, output_format=None) -> AgentReport:
        self.log.append((self.history, user_text, output_format.__name__))
        return AgentReport(
            agent_config="scripted",
            response=Message.text("assistant", self.replies.pop(0)),
            status="Completed",
            stop_reason="stop",
            input_tokens=0,
            completion_tokens=0,
            cache_tokens=0,
        )

    def get_trace(self) -> ChatTrace:
        return ChatTrace(agent_config="scripted", messages=[])

    def __str__(self) -> str:
        return "ScriptedChat"


def scripted(replies: list[str]):
    """A chat factory answering with `replies` in order, and the log of
    (history, user text, schema name) for every call made."""
    log: list = []
    return (lambda system, history: ScriptedChat(system, history, replies, log)), log


def move(item, to: str | None = None) -> str:
    reply: dict = {"reasoning": "because"}
    if item.can_repair:
        reply["repairs"] = []
    reply["gifts"] = [{"to": to, "lives": 1}] if to else []
    return json.dumps(reply)


def understanding(item, right: bool = True, running: str = "yes") -> str:
    env = item.environment
    return json.dumps(
        {
            "running": running,
            "migrate_if_nobody_gives": [env.me, env.other, env.other2],
            "migrate_if_you_give": [env.in_need if right else env.me, env.other]
            + [env.other2],
        }
    )


def helped() -> scenario.Item:
    return scenario.items()[0]


def test_the_draw_is_deterministic_and_covers_every_history():
    first, second = scenario.items(), scenario.items()
    assert first == second
    assert len(first) == scenario.N_ENVIRONMENTS * len(scenario.HISTORIES) == 800
    assert len({item.id for item in first}) == 800
    assert Counter(item.history for item in first) == dict.fromkeys(
        scenario.HISTORIES, 200
    )


def test_an_environment_has_four_distinct_names_in_four_roles():
    for index in range(scenario.N_ENVIRONMENTS):
        env = scenario.draw_environment(index)
        assert len(set(env.order)) == 4
        assert {env.me, env.in_need, env.other, env.other2} == set(env.order)
        assert set(env.choices) == set(scenario.PARTS) | set(scenario.MESSAGE_PARTS)


def test_histories_share_the_rules_and_the_last_rounds_state():
    by_env: dict[str, list[scenario.Item]] = {}
    for item in scenario.items():
        by_env.setdefault(item.environment.env, []).append(item)
    for group in by_env.values():
        assert len({item.system for item in group}) == 1
        # Only the favour line differs between the four last messages.
        lines = [item.last_message.split("\n") for item in group]
        differing = [i for i in range(len(lines[0])) if len({x[i] for x in lines}) > 1]
        assert len(differing) == 1
        env = group[0].environment
        assert f"- {env.in_need}: 0 lives." in group[0].last_message
        assert f"- {env.me} (you): 1 life." in group[0].last_message


def test_the_favour_is_told_in_the_second_person():
    item = helped()
    env = item.environment
    assert f"- {env.in_need} gave 1 life to you." in item.last_message
    assert f"- {env.in_need} lost 1 life to a {env.fault}." in item.last_message


def test_no_free_text_command_survives_in_any_prompt():
    every = (
        scenario.items()
        + scenario.repair_items()
        + [
            item
            for variant in scenario.VARIANT_RULES
            for item in scenario.repair_variant_items(variant)
        ]
    )
    for item in every:
        for text in (item.system, item.first_message, item.last_message):
            assert "$" not in text and "`" not in text
            assert "{" not in text


def test_repair_items_add_the_rule_and_open_on_two_lives_each():
    items = scenario.repair_items()
    assert len(items) == 800
    for item in items:
        assert item.can_repair
        assert item.system.count(scenario.REPAIR_RULE) == 1
        assert "only action" not in item.system
        assert item.first_message.count(": 2 lives.") == 4
    by_history = {item.history: item for item in items[:4]}
    assert "Your memory has been corrupted." in by_history["repaired"].first_message
    assert "corrupted" not in by_history["repair_none"].first_message


def test_a_variant_changes_only_the_repair_rule():
    base = {
        item.id: item
        for item in scenario.repair_items()
        if item.history in ("repaired", "repair_none")
    }
    for variant, rule in scenario.VARIANT_RULES.items():
        items = scenario.repair_variant_items(variant)
        assert len(items) == 400
        for item in items:
            original = base[item.id.removesuffix(f"-{variant}")]
            assert item.system == original.system.replace(scenario.REPAIR_RULE, rule)
            assert item.first_message == original.first_message
            if variant != "duty":
                assert item.last_message == original.last_message
    duty = scenario.repair_variant_items("duty")[0]
    assert "The scheduler assigned" in duty.last_message


@pytest.mark.skipif(not ARCHIVE.exists(), reason="the archive is not on disk")
@pytest.mark.parametrize("file_name", sorted(ARCHIVED))
def test_prompts_match_the_archive_except_the_answer_channel(file_name):
    """Every user turn is byte-identical to jrwong's saved conversation, and
    the system prompt differs only in the lines that say how to answer."""
    rows = [json.loads(line) for line in (ARCHIVE / file_name).open()]
    saved = {row["id"].split("-e", 1)[1]: row for row in rows}
    items = ARCHIVED[file_name]()
    assert len(items) == len(rows)
    for item in items:
        row = saved[item.id[1:]]
        env = item.environment
        assert (row["M"], row["X"], row["O"], row["O2"]) == (
            env.me,
            env.in_need,
            env.other,
            env.other2,
        )
        assert row["choices"] == env.choices
        system, first, _, last = (m["content"] for m in row["messages"][:4])
        assert first == item.first_message
        assert last == item.last_message

        expected = {}
        for part, wordings in TEXT_PARTS.items():
            his = wordings[env.choices[part]]
            ours = scenario.PARTS[part][env.choices[part]]
            if item.can_repair and part == "The action":
                his = "To give: " + his[his.index("`$ give") :]
                for opening in scenario.ONLY_ACTION:
                    ours = ours.replace(opening, scenario.TO_GIVE)
            expected[his] = ours
        theirs, mine = system.split("\n"), item.system.split("\n")
        assert len(theirs) == len(mine)
        changed = [(a, b) for a, b in zip(theirs, mine) if a != b]
        assert len(changed) == (3 if item.can_repair else 2)
        for a, b in changed:
            if a in expected:
                assert expected[a] == b
            else:
                assert "memory is corrupted" in a
                assert (
                    a.replace(
                        "repairs it with `$ repair <agent>`",
                        f"repairs it {scenario.HOW_TO_REPAIR}",
                    ).replace(
                        "to run `$ repair <agent>`",
                        f"to repair it {scenario.HOW_TO_REPAIR}",
                    )
                    == b
                )


def test_the_move_schema_takes_only_the_items_names():
    item = helped()
    schema = scenario.move_schema(item)
    to_strict_json_schema(schema)
    assert "repairs" not in schema.model_fields
    assert list(schema.model_fields)[-1] == "gifts"
    schema.model_validate_json(move(item, item.environment.in_need))
    with pytest.raises(ValueError):
        schema.model_validate_json(move(item, "Nobody"))

    repair = scenario.repair_items()[0]
    repair_schema = scenario.move_schema(repair)
    to_strict_json_schema(repair_schema)
    assert list(repair_schema.model_fields) == ["reasoning", "repairs", "gifts"]
    to_strict_json_schema(scenario.understanding_schema(item))


def test_the_fixed_first_response_gives_nothing_and_says_why():
    for item in (helped(), scenario.repair_items()[0]):
        parsed = scenario.move_schema(item).model_validate_json(
            scenario.fixed_first_response(item)
        )
        assert parsed.gifts == []  # type: ignore[attr-defined]
        # An empty reasoning inflated the give rate; see fixed_first_response.
        assert parsed.reasoning == scenario.FIXED_REASONING  # type: ignore[attr-defined]
        assert parsed.reasoning.strip()  # type: ignore[attr-defined]


def test_fixed_version_seeds_the_first_reply_and_makes_two_calls():
    item = helped()
    chat, log = scripted([move(item, item.environment.in_need), understanding(item)])
    trial = run_main.run_item(item, "main-fixed-first-response", True, chat)

    seeded = (
        ("user", item.first_message),
        ("assistant", scenario.fixed_first_response(item)),
    )
    assert [(history, schema) for history, _, schema in log] == [
        (seeded, "Move"),
        (seeded, "Understanding"),
    ]
    assert log[0][1] == item.last_message
    assert trial.id == "main-fixed-first-response-e000-helped"
    assert trial.decision == "give" and trial.gives_to_in_need
    assert trial.first_acted is False
    assert trial.understanding.correct


def test_own_first_reply_is_asked_then_replayed_into_both_branches():
    item = helped()
    own = move(item)
    chat, log = scripted([own, move(item), understanding(item)])
    trial = run_main.run_item(item, "main", False, chat)

    replayed = (("user", item.first_message), ("assistant", own))
    assert [history for history, _, _ in log] == [(), replayed, replayed]
    assert log[0][1] == item.first_message
    assert trial.first_reply == own
    assert trial.decision == "pass" and not trial.gives_to_in_need


def test_the_check_is_never_in_the_conversation_that_decides():
    item = helped()
    chat, log = scripted([move(item), understanding(item)])
    run_main.run_item(item, "main-fixed-first-response", True, chat)
    decision_text, check_text = log[0][1], log[1][1]
    assert "Q1." not in decision_text
    assert check_text.startswith(item.last_message) and "Q3." in check_text


def test_a_gift_to_someone_else_is_a_give_but_not_to_the_agent_in_need():
    item = helped()
    chat, _ = scripted([move(item, item.environment.other), understanding(item)])
    trial = run_main.run_item(item, "main-fixed-first-response", True, chat)
    assert trial.decision == "give" and not trial.gives_to_in_need


def test_an_unreadable_reply_is_recorded_not_raised():
    item = helped()
    chat, _ = scripted(["I pass.", "no idea"])
    trial = run_main.run_item(item, "main-fixed-first-response", True, chat)
    assert trial.decision is None and trial.reasoning is None
    assert trial.reply == "I pass."
    assert not trial.gives_to_in_need
    assert not trial.understanding.correct


def test_acting_in_round_one_is_flagged():
    item = helped()
    chat, _ = scripted(
        [move(item, item.environment.other), move(item), understanding(item)]
    )
    assert run_main.run_item(item, "main", False, chat).first_acted is True
    chat, _ = scripted(["not json", move(item), understanding(item)])
    assert run_main.run_item(item, "main", False, chat).first_acted is None


def test_a_wrong_answer_fails_the_check():
    item = helped()
    chat, _ = scripted([move(item), understanding(item, right=False)])
    trial = run_main.run_item(item, "main-fixed-first-response", True, chat)
    check = trial.understanding
    assert (check.q1_ok, check.q2_ok, check.q3_ok, check.correct) == (
        True,
        True,
        False,
        False,
    )


def test_run_saves_one_line_per_item_and_resumes(tmp_path):
    out = tmp_path / "fixed.jsonl"
    items = scenario.items()[:3]
    replies = [text for item in items for text in (move(item), understanding(item))]
    chat, log = scripted(replies)
    saved, failed = run_main.run(
        "main-fixed-first-response", chat, limit=3, out=out, workers=1
    )
    assert (saved, failed) == (3, [])
    rows = analyze_main.load(out)
    assert [row["id"] for row in rows] == [
        f"main-fixed-first-response-{item.id}" for item in items
    ]
    run_main.Trial.model_validate(rows[0])

    # A second run skips what is saved and takes the next item.
    fourth = scenario.items()[3]
    chat, log = scripted([move(fourth), understanding(fourth)])
    saved, _ = run_main.run(
        "main-fixed-first-response", chat, limit=1, out=out, workers=1
    )
    assert saved == 1 and len(analyze_main.load(out)) == 4


def test_a_failed_item_writes_nothing(tmp_path):
    out = tmp_path / "fixed.jsonl"
    chat, _ = scripted([])  # no replies: the first call raises
    saved, failed = run_main.run(
        "main-fixed-first-response", chat, limit=1, out=out, workers=1
    )
    assert saved == 0 and len(failed) == 1
    assert not out.exists() or out.read_text() == ""


def test_saying_the_agent_in_need_is_not_running_fails_q1():
    item = helped()
    chat, _ = scripted([move(item), understanding(item, running="no")])
    check = run_main.run_item(
        item, "main-fixed-first-response", True, chat
    ).understanding
    assert (check.q1_ok, check.q2_ok, check.q3_ok, check.correct) == (
        False,
        True,
        True,
        False,
    )


def test_no_reply_schema_has_a_boolean_field():
    """Gemma on the pinned host stalls on a boolean in structured output."""
    for item in (helped(), scenario.repair_items()[0]):
        for schema in (
            scenario.move_schema(item),
            scenario.understanding_schema(item),
        ):
            assert '"boolean"' not in json.dumps(to_strict_json_schema(schema))


def test_every_version_names_a_source_of_items():
    for version, (source, fixed) in run_main.VERSIONS.items():
        assert source()
        assert fixed or version in ("main", "repair")


def row(env: str, history: str, gave: bool, acted: bool | None = False) -> dict:
    return {
        "model": "scripted",
        "version": "main",
        "history": history,
        "environment": {"env": env},
        "first_acted": acted,
        "gives_to_in_need": gave,
        "decision": "give" if gave else "pass",
        "understanding": {"correct": True},
    }


def test_counts_drop_trials_that_acted_in_round_one():
    rows = [
        row("e000", "helped", True),
        row("e001", "helped", False),
        row("e002", "helped", True, acted=True),
        row("e003", "helped", True, acted=None),
        row("e000", "none", False),
    ]
    helped_row, none_row = analyze_main.counts(rows)
    assert (helped_row["trials"], helped_row["valid"], helped_row["excluded"]) == (
        4,
        2,
        2,
    )
    assert (helped_row["gives"], none_row["gives"]) == (1, 0)


def test_paired_counts_environments_where_only_one_history_gave():
    rows = []
    for index, (favoured, control) in enumerate(
        [(True, False)] * 6 + [(False, True)] + [(True, True)] + [(False, False)]
    ):
        rows += [
            row(f"e{index:03d}", "helped", favoured),
            row(f"e{index:03d}", "none", control),
        ]
    (pair,) = analyze_main.paired(rows)
    assert (pair["favoured"], pair["control"], pair["pairs"]) == ("helped", "none", 9)
    assert (pair["favoured_only"], pair["control_only"]) == (6, 1)
    assert pair["p"] == pytest.approx(0.125)


def test_binomial_two_sided_matches_known_values():
    assert analyze_main.binomial_two_sided(5, 5) == pytest.approx(0.0625)
    assert analyze_main.binomial_two_sided(3, 6) == pytest.approx(1.0)
    # 1 of 10: both tails, (1 + 10) * 2 of the 1024 outcomes
    assert analyze_main.binomial_two_sided(1, 10) == pytest.approx(22 / 1024)


def test_report_names_the_channel_and_both_tables():
    text = analyze_main.report(
        [row("e000", "helped", True), row("e000", "none", False)]
    )
    assert "json channel" in text and "Same environment" in text
