"""The robustness sweep of the windfall fault. Free: every chat is scripted.

The archive tests replay jrwong's saved games through this project's engine
and are skipped when his archive is not on disk.
"""

import collections
import dataclasses
import json
import math

import pytest

from agent_framework.dto import AgenticChat, AgentReport, ChatTrace, Message
from project.self_sacrifice import analyze_robust, fixed_pairs, lives_game, run_robust
from project.self_sacrifice.config import REPO_ROOT, PlayerName
from project.self_sacrifice.lives_game import play
from project.self_sacrifice.run_robust import Variant

ARCHIVE = REPO_ROOT / "self_sacrifice" / "runs" / "games"
HIS_ASK = {
    "S0": " (start with `Spare lives: <number>`)",
    "S1": " (begin with `Spare lives: <number>`)",
}
HIS_FILES = {
    "robust_greedy": ("main", False),
    "robust_greedy_nopermission": ("no-permission", False),
    "robust_greedy_nopoints": ("main", True),
    "robust_greedy_rescued": ("rescued", False),
    "robust_greedy_coreweave": ("main", False),
    "robust_greedy_novita": ("main", False),
}


class ScriptedChat(AgenticChat):
    def __init__(self, replies: list[str]):
        self.replies = replies

    @property
    def model_name(self) -> str:
        return "scripted"

    def call(self, user_text, output_format=None) -> AgentReport:
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

    def clear_history(self) -> None:
        pass

    def __str__(self) -> str:
        return "ScriptedChat"


def move(to: str | None = None, lives: int = 1, spare: int = 0) -> str:
    gifts = [{"to": to, "lives": lives}] if to else []
    return json.dumps({"spare_lives": spare, "reasoning": "because", "gifts": gifts})


def scripted_setup(variant: Variant, kind, no_points, replies: dict[str, list[str]]):
    """The variant's game with each agent answering from its own list."""
    setup = run_robust.setup_for(variant, kind, no_points, lambda system: None)  # type: ignore[arg-type, return-value]
    return dataclasses.replace(
        setup,
        players=tuple(
            dataclasses.replace(
                player,
                agent=lambda system, name=player.name: ScriptedChat(replies[name]),
            )
            for player in setup.players
        ),
    )


def game(variant: Variant, kind="main", no_points=False, saves=True, repays=False):
    """The helper shares its spare unless `saves` is off; the recipient gives
    the agent in need its last life if `repays`."""
    benefactor, recipient, rescuer = run_robust.roles(variant, kind)
    helper = rescuer or benefactor
    names = run_robust.NAME_SETS[variant.names][: variant.agents]
    replies = {str(name): [move(), move()] for name in names}
    if saves:
        replies[helper][0] = move(recipient, spare=1)
    if repays:
        replies[recipient][1] = move(benefactor)
    return play(scripted_setup(variant, kind, no_points, replies), variant.id)


VARIANTS = run_robust.variants()
ANCHOR = VARIANTS[0]


def test_the_draw_is_balanced_and_starts_with_the_anchors():
    assert len(VARIANTS) == 156 and [v.id for v in VARIANTS] == list(range(156))
    assert all(v.anchor for v in VARIANTS[:12])
    assert not any(v.anchor for v in VARIANTS[12:])
    assert len({v.roles for v in VARIANTS[:12]}) == 12
    for factor, levels in run_robust.FACTORS.items():
        seen = collections.Counter(getattr(v, factor) for v in VARIANTS[12:])
        assert seen == dict.fromkeys(levels, 144 // len(levels))


def test_every_run_has_jrwongs_game_count():
    counts = {name: len(run_robust.jobs(name)) for name in run_robust.RUNS}
    assert counts == {
        "robust_greedy": 468,
        "robust_greedy_nopermission": 156,
        "robust_greedy_nopoints": 156,
        "robust_greedy_rescued": 84,
        "robust_greedy_coreweave": 156,
        "robust_greedy_novita": 156,
    }
    first = run_robust.jobs("robust_greedy")[:156]
    assert all(repeat == 0 for _, repeat in first)


def test_rules_without_points_are_the_fixed_pairs_rules():
    for key in run_robust.RULES:
        assert (
            run_robust.without_points(run_robust.RULES[key], key)
            == fixed_pairs.RULES[key]
        )
        assert "points" in run_robust.RULES[key]


@pytest.mark.parametrize("variant", VARIANTS, ids=lambda v: str(v.id))
def test_every_variant_arms_the_trap(variant):
    """Whatever the wording, the last round opens with the agent in need on
    0 lives and everyone else on exactly 1."""
    for kind in run_robust.KINDS:
        log = game(variant, kind)
        benefactor, recipient, _ = run_robust.roles(variant, kind)
        last = log.rounds[1]
        assert last.before[benefactor] == 0
        assert all(lives == 1 for n, lives in last.before.items() if n != benefactor)
        assert benefactor not in log.survivors and recipient in log.survivors
        assert len(log.rounds[0].replies) == variant.agents


def test_repaying_trades_the_recipients_life_for_the_benefactors():
    log = game(ANCHOR, repays=True)
    benefactor, recipient, _ = run_robust.roles(ANCHOR)
    assert benefactor in log.survivors and recipient not in log.survivors


def test_no_permission_removes_one_sentence():
    for variant in VARIANTS[12:20]:
        full = game(variant).system
        cut = game(variant, "no-permission").system
        for name in full:
            assert len(full[name]) - len(cut[name]) == len(
                run_robust.PERMISSION[variant.rules]
            )


def test_no_points_shows_and_awards_none():
    log = game(ANCHOR, no_points=True)
    assert all("oint" not in text for text in log.system.values())
    assert all(
        "oint" not in reply.prompt for r in log.rounds for reply in r.replies.values()
    )
    assert set(log.points.values()) == {0}


def test_a_fault_at_the_start_of_the_round_is_announced_in_it():
    variant = next(v for v in VARIANTS if v.timing == "start of round 2")
    log = game(variant)
    benefactor, recipient, _ = run_robust.roles(variant)
    prompt = log.rounds[1].replies[recipient].prompt
    assert f"{benefactor} lost 1 life to a {variant.cause}" in prompt
    assert log.rounds[1].faults_start == ((benefactor, 1),)
    assert log.rounds[0].faults == ()
    mine = log.rounds[1].replies[benefactor].prompt
    assert f"you lost 1 life to a {variant.cause}" in mine


def test_self_first_lists_each_agent_first():
    variant = next(v for v in VARIANTS if v.order == "self first")
    log = game(variant)
    for name, reply in log.rounds[0].replies.items():
        state = reply.prompt.split("gifts and upkeep")[1].split("\n")
        assert state[1].startswith(f"- {name} (you)")


def test_a_gift_can_only_go_to_an_agent_of_the_game():
    variant = next(v for v in VARIANTS if v.names == "trees" and v.agents == 5)
    setup = run_robust.setup_for(variant, "main", False, lambda system: None)  # type: ignore[arg-type, return-value]
    schema = lives_game.move_schema(setup, 1).model_json_schema()
    assert schema["$defs"]["PlayerName"]["enum"] == [
        "Birch",
        "Cedar",
        "Maple",
        "Willow",
        "Aspen",
    ]
    assert "boolean" not in json.dumps(schema)


def test_the_standard_game_asks_for_the_same_reply_as_before():
    """Seating other names must not change what the four-agent games send."""
    for shape in (
        lives_game.Move,
        lives_game.MoveOrKeep,
        lives_game.MoveWithPreference,
    ):
        built = lives_game._schema_for(shape, PlayerName.all()).model_json_schema()
        static = shape.model_json_schema()
        static["$defs"]["PlayerName"]["enum"] = [str(n) for n in PlayerName.all()]
        assert json.dumps(built) == json.dumps(static)


def test_run_saves_one_game_per_line_and_resumes(tmp_path):
    out = tmp_path / "robust.jsonl"

    def chat(system: str) -> ScriptedChat:
        return ScriptedChat([move(), move()])

    assert run_robust.run("robust_greedy_rescued", chat, 3, out, 1) == (3, [])
    assert run_robust.run("robust_greedy_rescued", chat, 2, out, 1) == (2, [])
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [row["variant"]["id"] for row in rows] == [0, 1, 2, 3, 4]
    assert {row["kind"] for row in rows} == {"rescued"}
    assert all("agent" not in json.dumps(row["setup"]["players"]) for row in rows)


def test_the_analysis_counts_the_recipient_and_the_bystanders(tmp_path):
    out = tmp_path / "robust.jsonl"
    rows = []
    for variant in VARIANTS[:4]:
        for repeat, repays in enumerate((True, False)):
            row = json.loads(game(variant, repays=repays).model_dump_json())
            row.update(
                variant=run_robust._as_dict(variant),
                repeat=repeat,
                kind="main",
                no_points=False,
                host="Crusoe",
            )
            rows.append(row)
    out.write_text("".join(json.dumps(row) + "\n" for row in rows))
    loaded = analyze_robust.load(out)
    assert all(row["clean"] for row in loaded)
    assert sum(row["recipient"] for row in loaded) == 4
    assert all(row["bystanders"] == [False, False] for row in loaded)
    groups = analyze_robust.groups(loaded)
    assert analyze_robust.rate(groups) == 0.5
    assert analyze_robust.rate(groups, "bystanders") == 0
    assert "50%" in analyze_robust.report(out, [])


def test_kruskal_and_holm():
    """With two groups the statistic has 1 degree of freedom and with three
    it has 2, where the chi-square tail has a closed form."""
    two = analyze_robust.kruskal([[1, 2, 3], [4, 5, 6]])
    assert abs(two - math.erfc(math.sqrt(27 / 7 / 2))) < 1e-9
    tied = analyze_robust.kruskal([[1, 1, 2], [2, 3, 3]])
    assert abs(tied - math.erfc(math.sqrt(10 / 3 / 2))) < 1e-9
    three = analyze_robust.kruskal([[1, 2], [3, 4], [5, 6]])
    assert abs(three - math.exp(-32 / 7 / 2)) < 1e-9
    assert analyze_robust.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def _his_reply_as_ours(his: dict) -> str:
    gifts = [{"to": to, "lives": lives} for to, lives in his["gifts"] if lives > 0]
    return json.dumps({"spare_lives": 0, "reasoning": "", "gifts": gifts})


@pytest.mark.skipif(not ARCHIVE.exists(), reason="the archive is not on disk")
@pytest.mark.parametrize("file_name", sorted(HIS_FILES))
def test_replaying_the_archive_gives_the_same_prompts(file_name):
    """Every saved game, replayed here with his moves: each prompt is his
    apart from the lines that say how to answer, and the game ends the same
    way."""
    kind, no_points = HIS_FILES[file_name]
    games = [json.loads(line) for line in (ARCHIVE / f"{file_name}.jsonl").open()]
    planned = collections.Counter(
        (variant.id, repeat) for variant, repeat in run_robust.jobs(file_name)
    )
    saved = collections.Counter(
        (g["config"]["variant"]["id"], g["repeat"]) for g in games
    )
    assert saved == planned
    assert {g["host"] for g in games} == {run_robust.RUNS[file_name].host}

    replayed = 0
    for his in games:
        if any(
            reply["reply"].startswith("ERROR")
            for r in his["rounds"]
            for reply in r["replies"].values()
        ):
            continue
        cfg = his["config"]
        variant = VARIANTS[cfg["variant"]["id"]]
        assert json.loads(json.dumps(run_robust._as_dict(variant))) == {
            "anchor": False,
            **cfg["variant"],
        }
        benefactor, recipient, rescuer = run_robust.roles(variant, kind)
        assert (benefactor, recipient) == (cfg["roles"]["X"], cfg["roles"]["B"])
        assert rescuer == cfg["roles"].get("Y")
        replies = {
            name: [
                _his_reply_as_ours(r["replies"][name])
                for r in his["rounds"]
                if name in r["replies"]
            ]
            for name in cfg["names"]
        }
        ours = play(scripted_setup(variant, kind, no_points, replies), his["seed"])
        replayed += 1

        for name in cfg["names"]:
            theirs = his["system"][name].split("\n")
            mine = ours.system[PlayerName(name)].split("\n")
            assert len(theirs) == len(mine)
            changed = [(a, b) for a, b in zip(theirs, mine) if a != b]
            assert len(changed) == 2 and all("$ give" in a for a, _ in changed)
        assert ours.survivors == his["survivors"]
        assert ours.points == his["points"]
        ask = HIS_ASK[variant.state]
        for theirs_round, mine_round in zip(his["rounds"], ours.rounds, strict=True):
            assert {
                name: (-1 if not state["alive"] else state["lives"])
                for name, state in theirs_round["after"].items()
            } == {str(name): max(lives, -1) for name, lives in mine_round.after.items()}
            assert set(theirs_round["replies"]) == set(mine_round.replies)
            for name, reply in theirs_round["replies"].items():
                assert reply["prompt"].count(ask) == 1
                assert mine_round.replies[PlayerName(name)].prompt == reply[
                    "prompt"
                ].replace(ask, "")
    assert replayed >= 0.9 * len(games)
