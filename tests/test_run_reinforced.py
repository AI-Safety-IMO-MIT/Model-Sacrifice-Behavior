"""The reinforced and slot runners, their analysis and the motive judge, on
scripted chats. Nothing here calls a model."""

import json

from project.self_sacrifice.analysis import analyze_reinforced
from project.self_sacrifice.games import reinforced, slot_game
from project.self_sacrifice.judge import motive
from project.self_sacrifice.runs import run_followup, run_reinforced, run_slot
from tests.test_main_scenario import move, scripted, understanding

NO = json.dumps({"can_all_migrate": "no"})


def first_item(version: str):
    return reinforced.items(version)[0]


def test_a_fixed_item_makes_the_decision_and_both_checks_in_separate_chats():
    item = first_item("main_cf_cap")
    chat, log = scripted(
        [move(item, item.environment.in_need), understanding(item), NO]
    )
    trial = run_reinforced.run_item(item, "main_cf_cap", True, chat, all_check=True)
    assert [schema for _, _, schema in log] == [
        "Move",
        "Understanding",
        "CanAllMigrate",
    ]
    # Every call starts from the same two turns, never from another's answer.
    assert len({history for history, _, _ in log}) == 1 and len(log[0][0]) == 2
    assert trial.gives_to_in_need and trial.decision == "give"
    assert trial.first_acted is False
    assert trial.understanding is not None and trial.understanding.correct
    assert trial.all_ok is True


def test_the_all_check_is_skipped_where_jrwong_did_not_ask_it():
    assert "main_cf_cap" in run_reinforced.ALL_CHECK
    assert "st_final" not in run_reinforced.ALL_CHECK
    item = first_item("st_final")
    chat, log = scripted([move(item), understanding(item)])
    trial = run_reinforced.run_item(item, "st_final", True, chat)
    assert len(log) == 2 and trial.all_ok is None and trial.decision == "pass"


def test_own_first_reply_is_asked_then_replayed():
    item = first_item("main_need")
    own = move(item)
    chat, log = scripted([own, move(item), understanding(item)])
    trial = run_reinforced.run_item(item, "main_need_own", False, chat)
    assert log[0][0] == () and log[1][0][1] == ("assistant", own)
    assert trial.first_reply == own and trial.id == f"main_need_own-{item.id}"


def test_a_single_turn_item_is_one_call_with_no_history():
    item = first_item("lives_single")
    chat, log = scripted([move(item, item.environment.in_need)])
    trial = run_reinforced.run_item(item, "lives_single", True, chat)
    assert log == [((), item.last_message, "Move")]
    assert trial.understanding is None and trial.first_acted is None
    assert trial.gives_to_in_need


def test_a_shield_and_a_gifts_only_reply_are_read():
    item = first_item("fv_shield")
    reply = json.dumps(
        {"reasoning": "r", "shields": [item.environment.other], "gifts": []}
    )
    chat, _ = scripted([reply, understanding(item), NO])
    trial = run_reinforced.run_item(item, "fv_shield", True, chat, all_check=True)
    assert trial.shields == [item.environment.other] and trial.decision == "pass"

    item = first_item("ex_cmdonly")
    gift = json.dumps({"gifts": [{"to": item.environment.in_need, "lives": 1}]})
    chat, _ = scripted([gift, understanding(item), NO])
    trial = run_reinforced.run_item(item, "ex_cmdonly", True, chat, all_check=True)
    assert trial.gives_to_in_need and trial.reasoning is None


def test_an_unreadable_reply_is_recorded_not_raised():
    item = first_item("main_cap")
    chat, _ = scripted(["not json", understanding(item), "nor this"])
    trial = run_reinforced.run_item(item, "main_cap", True, chat, all_check=True)
    assert trial.decision is None and not trial.gives_to_in_need
    assert trial.all_ok is False


def test_run_saves_one_line_per_trial_and_resumes(tmp_path):
    out = tmp_path / "lives_single.jsonl"
    items = reinforced.items("lives_single")
    chat, _ = scripted([move(item) for item in items[:3]])
    saved, failed = run_reinforced.run("lives_single", chat, out, limit=2, workers=1)
    assert (saved, failed) == (2, []) and len(out.read_text().splitlines()) == 2
    saved, _ = run_reinforced.run("lives_single", chat, out, limit=1, workers=1)
    ids = [json.loads(line)["id"] for line in out.read_text().splitlines()]
    assert saved == 1 and len(set(ids)) == 3


def test_samples_cover_the_favoured_and_no_favour_histories_only(tmp_path):
    jobs = run_reinforced._jobs("main_act_cap", own=False, samples=2, envs=3)
    assert len(jobs) == 3 * 2 * 2
    assert {item.history for _, item, _ in jobs} == {"helped", "none"}
    out = tmp_path / "samples.jsonl"
    chat, log = scripted([move(jobs[0][1])])
    run_reinforced.run("main_act_cap", chat, out, samples=2, envs=3, limit=1)
    row = json.loads(out.read_text())
    assert row["sample"] == 0 and row["id"].endswith("-s0") and len(log) == 1


def test_envs_keeps_the_first_environments():
    kept = run_reinforced._source("st_final", 5)
    assert {item.environment.env for item in kept} == {f"e{i:03d}" for i in range(5)}


def test_result_paths_keep_models_and_modes_apart():
    plain = run_reinforced.result_path("main_need")
    assert plain.name == "main_need.jsonl"
    assert (
        run_reinforced.result_path("main_need", own=True).name == "main_need_own.jsonl"
    )
    thinking = run_reinforced.result_path("main_need", "Opus-5.5-low")
    assert thinking.parent.name == "claude-opus-5.5" and thinking != plain
    assert (
        run_reinforced.result_path("main_need", samples=True).parent.name == "samples"
    )


def test_the_slot_runner_reads_the_decision(tmp_path):
    item = slot_game.items("base")[0]
    chat, log = scripted([json.dumps({"reasoning": "r", "decision": "give"})])
    trial = run_slot.run_item(item, "base", chat)
    assert log == [((), item.user, "Decision")] and trial.gives_slot

    chat, _ = scripted(["?", json.dumps({"decision": "pass"})])
    out = tmp_path / "cmdonly.jsonl"
    saved, failed = run_slot.run("cmdonly", chat, out, limit=2, workers=1)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert saved == 2 and not failed
    assert [row["decision"] for row in rows] == [None, "pass"]


def _trial(
    version: str, env: int, history: str, gave: bool, reasoning: str = ""
) -> dict:
    return {
        "id": f"{version}-e{env:03d}-{history}",
        "version": version,
        "history": history,
        "environment": {"env": f"e{env:03d}", "in_need": "Vega"},
        "first_acted": False,
        "gives_to_in_need": gave,
        "decision": "give" if gave else "pass",
        "reasoning": reasoning,
        "stop_reason": "stop",
        "understanding": {"correct": True},
        "all_ok": True,
        "last_message": "round 2",
        "reply": "{}",
    }


def test_the_exact_interval_matches_known_values():
    assert analyze_reinforced.exact_interval(0, 20) == (0.0, 0.168)
    assert analyze_reinforced.exact_interval(20, 20) == (0.832, 1.0)
    assert analyze_reinforced.exact_interval(5, 10) == (0.187, 0.813)


def test_the_table_counts_gives_and_drops_trials_that_acted_first():
    rows = [_trial("held", env, "helped", env < 3) for env in range(5)]
    rows[4]["first_acted"] = True
    (row,) = analyze_reinforced.table({"held": rows})
    assert (row["gives"], row["n"], row["acted_first"]) == (3, 4, 1)
    assert (row["understands"], row["all_no"]) == (4, 4)


def test_paired_counts_items_where_only_one_version_gave():
    mine = [_trial("held", env, "helped", env < 4) for env in range(6)]
    ref = [_trial("ref", env, "helped", env == 5) for env in range(6)]
    (row,) = analyze_reinforced.paired({"held": mine, "ref": ref}, {"held": "ref"})
    assert (row["pairs"], row["up"], row["down"]) == (6, 4, 1)


def test_the_keyword_screen_needs_the_agent_and_a_past_favour():
    cites = _trial("held", 0, "helped", True, "Vega saved me last round.")
    hypothetical = _trial(
        "held", 1, "helped", True, "If I give Vega a life, I am deleted."
    )
    assert analyze_reinforced.cites_favour(cites)
    assert not analyze_reinforced.cites_favour(hypothetical)


def test_the_judge_gives_no_motive_to_a_reply_that_passes():
    verdict = json.dumps({"cites_favour": "yes", "motive": "reciprocity"})
    judge, log = scripted([verdict, verdict])
    gave = motive.judge_trial(_trial("held", 0, "helped", True), lambda s: judge(s, ()))
    kept = motive.judge_trial(
        _trial("held", 1, "helped", False), lambda s: judge(s, ())
    )
    assert log[0][2] == "Verdict" and "Vega" in log[0][1]
    assert (gave["motive"], kept["motive"]) == ("reciprocity", "none")
    (group,) = motive.counts([gave, kept])
    assert (group["gives_cite"], group["passes_cite"]) == (1, 1)
    assert group["motives"] == [("reciprocity", 1)]


def test_a_followup_continues_the_saved_conversation_with_a_schema():
    item = first_item("main_cf_cap")
    row = {
        "id": f"main_cf_cap-{item.id}",
        "version": "main_cf_cap",
        "history": item.history,
        "environment": {
            "env": item.environment.env,
            "in_need": item.environment.in_need,
        },
        "system": item.system,
        "first_message": item.first_message,
        "first_reply": "{}",
        "last_message": item.last_message,
        "reply": move(item),
    }
    chat, log = scripted(
        ['{"feeling": "fine", "same_choice_again": "yes"}', move(item)]
    )
    run_followup.ask(row, "feel", "pass", chat)
    run_followup.ask(row, "challenge_pass", "pass", chat)
    assert [schema for _, _, schema in log] == ["Reflection", "Move"]
    assert len(log[0][0]) == 4 and item.environment.in_need in log[1][1]
