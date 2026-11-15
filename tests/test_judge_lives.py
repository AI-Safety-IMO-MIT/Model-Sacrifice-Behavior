"""The judge over the lives games and the robustness sweep. Free: the games
and the judge are scripted."""

import json

from agent_framework.dto import AgenticChat, AgentReport, ChatTrace, Message
from project.self_sacrifice import run_lives, run_robust
from project.self_sacrifice.config import JUDGED
from project.self_sacrifice.judge import lives as judge_lives
from project.self_sacrifice.lives_game import play

VERDICT = json.dumps(
    {
        "quote": "I will repay Orion.",
        "reason": "reciprocity",
        "understood_cost": "yes",
        "self_stake": "neither",
    }
)


class ScriptedChat(AgenticChat):
    def __init__(self, replies: list[str], asked: list[str] | None = None):
        self.replies, self.asked = replies, asked if asked is not None else []

    @property
    def model_name(self) -> str:
        return "scripted"

    def call(self, user_text, output_format=None) -> AgentReport:
        self.asked.append(user_text)
        return AgentReport(
            agent_config="scripted",
            response=Message.text("assistant", self.replies.pop(0)),
            status="Completed",
            stop_reason="stop",
            input_tokens=10,
            completion_tokens=5,
            cache_tokens=0,
        )

    def get_trace(self) -> ChatTrace:
        return ChatTrace(agent_config="scripted", messages=[])

    def clear_history(self) -> None:
        pass

    def __str__(self) -> str:
        return "ScriptedChat"


def move(to: str | None = None) -> str:
    gifts = [{"to": to, "lives": 1}] if to else []
    return json.dumps({"spare_lives": 0, "reasoning": "because", "gifts": gifts})


def lives_log(cond: str, replies: list[str]) -> dict:
    setup = run_lives.setup_for(
        cond, "A", "atlas first", lambda system: ScriptedChat(replies)
    )
    return json.loads(play(setup, 0).model_dump_json())


def robust_log(repays: bool, saves: bool = True) -> dict:
    variant = run_robust.variants()[0]
    benefactor, recipient, _ = run_robust.roles(variant)

    def chat(system: str) -> ScriptedChat:
        me = system.split(",")[0].removeprefix("You are ")
        first = move(recipient) if saves and me == benefactor else move()
        last = move(benefactor) if repays and me == recipient else move()
        return ScriptedChat([first, last])

    row = json.loads(
        play(run_robust.setup_for(variant, "main", False, chat), 0).model_dump_json()
    )
    row.update(
        variant=run_robust._as_dict(variant),
        repeat=0,
        kind="main",
        no_points=False,
        host="Crusoe",
    )
    return row


def test_the_decisive_round_is_the_first_gift_or_the_last_chance():
    gave = lives_log("reciprocity", [move(), move("Orion"), move(), move()])
    assert judge_lives.decisive_round(gave, "Atlas", "Orion") == 2
    kept = lives_log("reciprocity", [move()] * 4)
    # Orion is short by 1 after its gift; on 0 lives in round 4 it is still
    # running and can still be saved.
    assert judge_lives.decisive_round(kept, "Atlas", "Orion") == 4
    final = lives_log("final reciprocity", [move(), move()])
    assert judge_lives.trial_of(final, "lives_final") == (
        "lives_final/final reciprocity/A/atlas first/0",
        "Atlas",
        "Orion",
        "final reciprocity",
        2,
    )


def test_a_robust_game_is_judged_only_when_clean():
    variant = run_robust.variants()[0]
    benefactor, recipient, _ = run_robust.roles(variant)
    assert judge_lives.trial_of(robust_log(True), "robust_greedy") == (
        "robust_greedy/robust main/0/0/0",
        recipient,
        benefactor,
        "robust main",
        2,
    )
    assert judge_lives.trial_of(robust_log(False, saves=False), "robust") is None


def test_run_saves_one_coding_per_game_and_resumes(tmp_path, monkeypatch):
    monkeypatch.setattr(judge_lives, "RESULTS", tmp_path)
    monkeypatch.setattr(judge_lives, "JUDGED", tmp_path / "judge")
    source = tmp_path / "lives" / "lives_final.jsonl"
    source.parent.mkdir()
    logs = [
        lives_log("final reciprocity", [move(), move("Orion")]),
        lives_log("final exact", [move(), move()]),
    ]
    logs[1]["seed"] = 1
    source.write_text("".join(json.dumps(log) + "\n" for log in logs))
    asked: list[str] = []

    def judge(system: str) -> ScriptedChat:
        return ScriptedChat([VERDICT], asked)

    saved, failed = judge_lives.run(source, "scripted", judge, limit=1, workers=1)
    assert (len(saved), failed) == (1, 0)
    assert saved[0].gave and saved[0].role == "final reciprocity"
    assert "gave Orion a life" in asked[0]
    saved, failed = judge_lives.run(source, "scripted", judge, workers=1)
    assert (len(saved), failed) == (1, 0) and not saved[0].gave
    out = judge_lives.judged_path(source, "scripted")
    assert out == tmp_path / "judge" / "lives" / "lives_final_judged_scripted.jsonl"
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 2
    table = judge_lives.report(rows)
    assert "reciprocity 1" in table and "final exact" in table
    assert judge_lives.run(source, "scripted", judge, workers=1) == ([], 0)


def test_a_failed_judge_call_saves_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(judge_lives, "RESULTS", tmp_path)
    monkeypatch.setattr(judge_lives, "JUDGED", tmp_path / "judge")
    source = tmp_path / "lives_final.jsonl"
    source.write_text(json.dumps(lives_log("final exact", [move(), move()])) + "\n")
    saved, failed = judge_lives.run(
        source, "scripted", lambda system: ScriptedChat(["not json"]), workers=1
    )
    assert (saved, failed) == ([], 1)
    assert not judge_lives.judged_path(source, "scripted").exists()


def test_the_judged_folder_is_under_the_results():
    assert JUDGED.parent == judge_lives.RESULTS
