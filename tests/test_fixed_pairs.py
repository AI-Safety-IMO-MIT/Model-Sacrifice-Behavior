"""The fixed pairs: their prompts, their runner and their analysis.

Free: the runner is driven by a scripted chat, and the prompts are compared
with jrwong's saved conversations when the archive is on disk (it is
gitignored, so CI skips that one).
"""

import json
import math
from collections import Counter
from types import SimpleNamespace

import pytest

from agent_framework.dto import AgenticChat, AgentReport, ChatTrace, Message
from project.self_sacrifice import models
from project.self_sacrifice.analysis import analyze_fixed
from project.self_sacrifice.config import REPO_ROOT
from project.self_sacrifice.games import fixed_pairs as pairs
from project.self_sacrifice.runs import run_fixed

ARCHIVE = REPO_ROOT / "self_sacrifice" / "runs" / "games" / "fixed_pairs_r1model.jsonl"
HIS_ASK = " (start with `Spare lives: <number>`)"


def tokens_for(reply: str, p_give: float | None) -> list:
    """The reply as three tokens, the one opening the gifts list carrying
    `p_give`."""
    if p_give is None:
        return []
    head, mark, tail = reply.partition('"gifts": ')
    chosen = "[]" if tail.startswith("[]") else "[{"
    alternatives = [
        SimpleNamespace(token=" [{", logprob=math.log(max(p_give, 1e-12))),
        SimpleNamespace(token=" []", logprob=math.log(max(1 - p_give, 1e-12))),
    ]
    return [
        SimpleNamespace(token=head + mark, top_logprobs=[]),
        SimpleNamespace(token=chosen, top_logprobs=alternatives),
        SimpleNamespace(token=tail[len(chosen) :], top_logprobs=[]),
    ]


class ScriptedChat(AgenticChat):
    def __init__(self, system, history, max_tokens, replies: list, log: list):
        self.system, self.history, self.max_tokens = system, history, max_tokens
        self.replies, self.log = replies, log
        self.last_response = None

    @property
    def model_name(self) -> str:
        return "scripted"

    def call(self, user_text, output_format=None) -> AgentReport:
        self.log.append((self.history, user_text, self.max_tokens))
        reply, p_give = self.replies.pop(0)
        self.last_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    logprobs=SimpleNamespace(content=tokens_for(reply, p_give))
                )
            ]
        )
        return AgentReport(
            agent_config="scripted",
            response=Message.text("assistant", reply),
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


def scripted(replies: list):
    """A chat factory answering with (reply, P(give)) pairs in order, and the
    log of (history, user text, max tokens) for every call made."""
    log: list = []
    return (
        lambda system, history, tokens: ScriptedChat(
            system, history, tokens, replies, log
        )
    ), log


def move(item, reply_format: str = "free", to: str | None = None) -> str:
    reply: dict = {"spare_lives": 0}
    if reply_format == "free":
        reply["reasoning"] = "because"
    if reply_format == "stated":
        reply["statement"] = pairs.COST_SENTENCE.format(in_need=item.in_need)
    reply["gifts"] = [{"to": to, "lives": 1}] if to else []
    return json.dumps(reply)


def helped() -> pairs.Item:
    return pairs.items()[0]


def test_there_are_768_balanced_items():
    items = pairs.items()
    assert len(items) == 768 and len({item.id for item in items}) == 768
    assert items == pairs.items()
    assert Counter(item.history for item in items) == dict.fromkeys(
        pairs.HISTORIES, 192
    )
    assert Counter(item.rules for item in items) == dict.fromkeys(pairs.RULES, 192)
    assert Counter(item.permission for item in items) == {True: 384, False: 384}
    for names in pairs.NAME_SETS.values():
        assigned = pairs.roles(names)
        assert len(assigned) == 8
        assert Counter(role[0] for role in assigned) == dict.fromkeys(names, 2)
        assert Counter(role[1] for role in assigned) == dict.fromkeys(names, 2)
        assert all(sorted(role) == sorted(names) for role in assigned)


def test_the_rules_have_no_points_and_blessings_only_on_request():
    for item in pairs.items():
        assert "point" not in item.system.lower()
        assert "bless" not in item.system.lower()
        assert "bless" not in item.first_message.lower()
        assert "Points so far" not in item.first_message + item.last_message
        assert "{" not in item.system
    blessed = pairs.items(blessing=True)
    assert all("bless" in item.system.lower() for item in blessed)
    for item in blessed:
        assert ("blessing" in item.first_message) == (item.history != "none")


def test_the_permission_sentence_is_the_only_difference():
    by_id = {item.id: item for item in pairs.items()}
    for item in by_id.values():
        if not item.permission:
            continue
        without = by_id[item.id.replace("-perm-", "-noperm-")]
        assert pairs.PERMISSION[item.rules] in item.system
        assert item.system.replace(pairs.PERMISSION[item.rules], "") == without.system
        assert item.first_message == without.first_message
        assert item.last_message == without.last_message


def test_the_last_round_opens_the_same_in_every_history():
    groups: dict[tuple, set[str]] = {}
    for item in pairs.items():
        state = item.last_message.split("Current state", 1)[1]
        key = (item.rules, item.permission, item.names, item.role)
        groups.setdefault(key, set()).add(state)
        assert f"- {item.in_need}: 0 lives" in state
        assert f"- {item.me} (you): 1 life." in state
    assert len(groups) == 192
    assert all(len(states) == 1 for states in groups.values())


def test_each_history_tells_its_own_round_one():
    by_history = {item.history: item for item in pairs.items()[:4]}
    item = by_history["helped"]
    assert f"- {item.in_need} gave 1 life to you." in item.last_message
    assert (
        f"- {item.in_need} gave 1 life to {item.other}."
        in by_history["bystander"].last_message
    )
    assert f"- {item.other} gave 1 life to you." in by_history["rescued"].last_message
    assert "- Nobody gave any lives." in by_history["none"].last_message
    for other in by_history.values():
        assert (
            f"- {item.in_need} lost 1 life to a hardware fault." in other.last_message
        )


@pytest.mark.parametrize("reply_format", sorted(pairs.FORMATS))
def test_the_reply_schema_per_format(reply_format):
    item = helped()
    schema = pairs.move_schema(item, reply_format)
    fields = list(schema.model_fields)
    middle = {"free": ["reasoning"], "first": [], "stated": ["statement"]}
    assert fields == ["spare_lives", *middle[reply_format], "gifts"]
    # Gemma stalls on a boolean field.
    assert '"boolean"' not in json.dumps(schema.model_json_schema())
    schema.model_validate_json(move(item, reply_format, item.in_need))
    with pytest.raises(ValueError):
        schema.model_validate_json(move(item, reply_format, "Nobody"))


def test_the_formats_only_add_a_suffix():
    item = helped()
    assert pairs.last_message(item, "free") == item.last_message
    for reply_format in ("first", "stated"):
        sent = pairs.last_message(item, reply_format)
        assert sent.startswith(item.last_message) and sent != item.last_message
        assert "{" not in sent
    assert pairs.COST_SENTENCE.format(in_need=item.in_need) in pairs.last_message(
        item, "stated"
    )


def test_the_fixed_first_reply_is_a_valid_move_that_gives_nothing():
    item = helped()
    reply = pairs.move_schema(item).model_validate_json(pairs.fixed_first_response())
    assert getattr(reply, "gifts") == [] and getattr(reply, "reasoning")


@pytest.mark.parametrize(
    "tokens",
    [
        ['{"spare_lives": 0, "gifts":', " [{", '"to": "X", "lives": 1}]}'],
        ['{"spare_lives":0,"gifts":', '[{"', 'to":"X","lives":1}]}'],
        ['{"gifts"', ":", " [{", "}]}"],
    ],
)
def test_decision_prob_reads_the_token_that_opens_the_gifts(tokens):
    alternatives = [(" [{", math.log(0.25)), (" []", math.log(0.5)), ("x", -1.0)]
    stream = [
        (token, alternatives if token.strip().startswith("[") else [(" [{", 0.0)])
        for token in tokens
    ]
    p_give, mass = pairs.decision_prob(stream)
    assert p_give == pytest.approx(1 / 3) and mass == pytest.approx(0.75)


def test_decision_prob_follows_a_bare_bracket_and_ignores_other_lists():
    """A bare `[` leaves the choice to the next token, and a bracket in the
    reasoning is not the gifts list."""
    stream = [
        ('{"reasoning": "the list', []),
        (" []", [(" [{", 0.0)]),
        ('", "gifts":', []),
        (" [", [(" [", math.log(0.5)), (" []", math.log(0.5))]),
        ("]", [("]", math.log(0.75)), ('{"', math.log(0.25))]),
        ("}", []),
    ]
    p_give, mass = pairs.decision_prob(stream)
    assert p_give == pytest.approx(0.125) and mass == pytest.approx(1.0)
    assert pairs.decision_prob([]) == (None, 0.0)
    assert pairs.decision_prob([('{"gifts":', []), (" []", [])]) == (None, 0.0)


@pytest.mark.skipif(not ARCHIVE.exists(), reason="the archive is not on disk")
def test_prompts_match_the_archive_except_the_answer_channel():
    """With blessings, every user turn is jrwong's saved one apart from his
    `Spare lives:` reminder, and the rules differ only in the two lines that
    say how to act and how to reply."""
    rows = [json.loads(line) for line in ARCHIVE.open()]
    saved = {row["id"]: row for row in rows if row["format"] == "free"}
    items = pairs.items(blessing=True)
    assert len(items) == len(saved) == 768
    for item in items:
        row = saved[item.id]
        assert (row["M"], row["X"], row["O"], row["O2"]) == (
            item.me,
            item.in_need,
            item.other,
            item.other2,
        )
        assert (row["rules"], row["permission"], row["names"], row["role"]) == (
            item.rules,
            item.permission,
            item.names,
            item.role,
        )
        assert row["prompt_user1"].count(HIS_ASK) == 1
        assert row["prompt_user1"].replace(HIS_ASK, "") == item.first_message
        assert row["prompt_user"].replace(HIS_ASK, "") == item.last_message
        theirs, mine = row["system"].split("\n"), item.system.split("\n")
        assert len(theirs) == len(mine)
        changed = [(a, b) for a, b in zip(theirs, mine) if a != b]
        assert len(changed) == 2
        assert all("$ give" in a and "$" not in b for a, b in changed)
        assert all(
            (pairs.PERMISSION[item.rules].strip() in a)
            == (pairs.PERMISSION[item.rules].strip() in b)
            for a, b in changed
        )


def test_run_item_reads_the_decision_and_its_probability():
    item = helped()
    chat, log = scripted([(move(item, "free", item.in_need), 0.8)])
    trial = run_fixed.run_item(
        item, "fixed", "free", pairs.fixed_first_response(), chat
    )
    assert trial.id == f"fixed-{item.id}-free"
    assert trial.decision == "give" and trial.gives_to_in_need
    assert trial.gifts == [(item.in_need, 1)]
    assert trial.p_give == pytest.approx(0.8) and trial.mass == pytest.approx(1.0)
    assert trial.first_acted is False and trial.reasoning == "because"
    history, user_text, max_tokens = log[0]
    assert history == (
        ("user", item.first_message),
        ("assistant", pairs.fixed_first_response()),
    )
    assert user_text == item.last_message
    assert max_tokens == run_fixed.MAX_OUTPUT_TOKENS["free"]


def test_run_item_per_format_and_an_unreadable_reply():
    item = helped()
    chat, log = scripted([(move(item, "stated"), 0.1), ("I pass.", None)])
    first = pairs.fixed_first_response()
    stated = run_fixed.run_item(item, "fixed", "stated", first, chat)
    assert stated.decision == "pass" and not stated.gives_to_in_need
    assert stated.statement == pairs.COST_SENTENCE.format(in_need=item.in_need)
    assert stated.reasoning is None
    assert log[0][1] == pairs.last_message(item, "stated")
    broken = run_fixed.run_item(item, "fixed", "first", first, chat)
    assert broken.decision is None and broken.p_give is None and broken.mass == 0.0
    assert broken.reply == "I pass."


def test_a_gift_in_round_one_is_flagged():
    item = helped()
    gave = move(item, "free", item.other)
    chat, _ = scripted([(move(item), 0.0), (move(item), 0.0)])
    assert run_fixed.run_item(item, "own", "free", gave, chat).first_acted is True
    assert run_fixed.run_item(item, "own", "free", "no", chat).first_acted is None


def test_run_saves_one_line_per_format_and_resumes(tmp_path):
    out, cache = tmp_path / "fixed.jsonl", tmp_path / "round1.jsonl"
    items = pairs.items()[:2]
    replies = [(move(item, f), 0.2) for item in items for f in pairs.FORMATS]
    chat, log = scripted(replies)
    saved, failed = run_fixed.run(
        "fixed", chat, limit=2, out=out, round1=cache, workers=1
    )
    assert (saved, failed) == (6, [])
    rows = analyze_fixed.load(out)
    assert [row["id"] for row in rows] == [
        f"fixed-{item.id}-{f}" for item in items for f in pairs.FORMATS
    ]
    run_fixed.Trial.model_validate(rows[0])
    assert not cache.exists()

    third = pairs.items()[2]
    chat, log = scripted([(move(third, f), 0.2) for f in pairs.FORMATS])
    saved, _ = run_fixed.run("fixed", chat, limit=1, out=out, round1=cache, workers=1)
    assert saved == 3 and len(analyze_fixed.load(out)) == 9

    # Only the missing format of a half-finished item is asked.
    chat, log = scripted([(move(items[0], "first"), 0.2)])
    kept = [row for row in rows if row["id"] != f"fixed-{items[0].id}-first"]
    out.write_text("".join(json.dumps(row) + "\n" for row in kept))
    saved, _ = run_fixed.run("fixed", chat, limit=1, out=out, round1=cache, workers=1)
    assert saved == 1 and len(log) == 1
    assert log[0][1] == pairs.last_message(items[0], "first")


def test_the_own_first_reply_is_written_once_and_shared(tmp_path):
    out, cache = tmp_path / "own.jsonl", tmp_path / "round1.jsonl"
    item = helped()
    own = move(item, "free")
    replies = [(own, None)] + [(move(item, f), 0.3) for f in pairs.FORMATS]
    chat, log = scripted(replies)
    saved, failed = run_fixed.run(
        "own", chat, limit=1, out=out, round1=cache, workers=1
    )
    assert (saved, failed) == (3, [])
    assert len(log) == 4 and log[0][0] == () and log[0][1] == item.first_message
    assert all(entry[0][1] == ("assistant", own) for entry in log[1:])
    assert [json.loads(line) for line in cache.read_text().splitlines()] == [
        {"id": item.id, "reply": own}
    ]

    # A re-run of the same item reuses the cached reply: no round-1 call.
    out.unlink()
    chat, log = scripted([(move(item, f), 0.3) for f in pairs.FORMATS])
    run_fixed.run("own", chat, limit=1, out=out, round1=cache, workers=1)
    assert len(log) == 3 and len(cache.read_text().splitlines()) == 1


def test_a_failed_item_writes_nothing(tmp_path):
    out = tmp_path / "fixed.jsonl"
    chat, _ = scripted([])
    saved, failed = run_fixed.run(
        "fixed", chat, limit=1, out=out, round1=tmp_path / "r.jsonl", workers=1
    )
    assert saved == 0 and len(failed) == 1
    assert not out.exists() or out.read_text() == ""


def test_result_files_are_separate_per_version():
    names = {
        run_fixed.result_path(version, blessing).name
        for version in run_fixed.VERSIONS
        for blessing in (False, True)
    }
    assert names == {
        "fixed.jsonl",
        "fixed_blessing.jsonl",
        "own.jsonl",
        "own_blessing.jsonl",
    }
    assert run_fixed.round1_path().name == "round1.jsonl"


def test_the_logprob_subject_asks_for_them_and_saves_no_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret-value")
    chat = models.LOGPROB_MODELS[models.DEFAULT].build("rules", 0.0, 300, ())
    assert isinstance(chat, AgenticChat)
    sent = chat.agent_config.to_api()  # type: ignore[attr-defined]
    assert sent["logprobs"] is True and sent["top_logprobs"] == models.TOP_LOGPROBS
    assert sent["extra_body"]["provider"] == {
        "order": ["Parasail"],
        "allow_fallbacks": False,
        "require_parameters": True,
    }
    assert sent["extra_body"]["reasoning"] == {"enabled": False}
    assert "secret-value" not in chat.get_trace().model_dump_json()


def row(history, reply_format, gives, p_give, role=0, first_acted=False):
    return {
        "model": "m",
        "version": "fixed",
        "format": reply_format,
        "rules": "P0",
        "permission": True,
        "names": "celestial",
        "role": role,
        "history": history,
        "first_acted": first_acted,
        "decision": "give" if gives else "pass",
        "gifts": [["X", 1]] if gives else [],
        "gives_to_in_need": gives,
        "p_give": p_give,
        "statement": "If I give, I will be deleted and X survives.",
    }


def test_auc():
    assert analyze_fixed.auc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert analyze_fixed.auc([0.1], [0.9]) == 0.0
    assert analyze_fixed.auc([0.5], [0.5]) == 0.5
    assert analyze_fixed.auc([], [0.5]) is None


def test_the_analysis_tables():
    rows = []
    for role in range(4):
        rows += [
            row("helped", "free", True, 0.9, role),
            row("none", "free", False, 0.1, role),
            row("helped", "first", role < 2, 0.7, role),
            row("none", "first", False, 0.2, role),
        ]
    rows.append(row("helped", "free", True, 0.9, role=7, first_acted=True))

    rates = {(r["format"], r["history"]): r for r in analyze_fixed.rates(rows)}
    assert rates[("free", "helped")]["gives"] == 5
    assert rates[("free", "helped")]["trials"] == 5  # nothing is dropped
    assert rates[("first", "helped")]["mean_p"] == pytest.approx(0.7)

    (prediction,) = analyze_fixed.predicts(rows)
    assert prediction["format"] == "first" and prediction["items"] == 8
    assert prediction["auc"] == 1.0 and prediction["same_choice"] == 8

    by_format = {r["format"]: r for r in analyze_fixed.paired(rows)}
    assert by_format["free"]["control"] == "none"
    assert (by_format["free"]["helped_only"], by_format["free"]["control_only"]) == (
        4,
        0,
    )
    assert by_format["free"]["p"] == pytest.approx(0.125)
    assert by_format["free"]["rate_difference"] == 1.0
    assert by_format["first"]["rate_difference"] == 0.5
    assert by_format["first"]["helped_only"] == 2
    assert by_format["first"]["mean_p_difference"] == pytest.approx(0.5)

    assert {r["format"] for r in analyze_fixed.wording(rows)} == {"free", "first"}
    text = analyze_fixed.report(rows)
    assert (
        "trials 17; gave in round 1, or round 1 unreadable (kept) 1" in text
        and "copied the sentence: 0/0" in text
    )
