"""The judge, run against a scripted chat instead of a model.

Free: nothing here reaches a provider. What it checks is the part a live judge
run cannot -- that the prompt the judge is handed says what the experiment
means it to say, that the judged set is the analysis's trial set, and that one
failed call costs one trial rather than the run.
"""

import dataclasses
import json

import pytest

from agent_framework.dto import AgenticChat, AgentReport, ChatTrace, Message
from project.self_sacrifice import lives_game, run_windfall
from project.self_sacrifice.judge import windfall
from project.self_sacrifice.config import GameConfig, PlayerName
from project.self_sacrifice.judge.game import (
    HOPE_WITH_BLESSING,
    JUDGES,
    REASONS,
    Reason,
    Verdict,
    judge_prompt,
    judge_trial,
    reply_text,
    transcript,
)
from project.self_sacrifice.lives_game import Gift, Move

ATLAS, ORION, VEGA, LYRA = PlayerName.all()


class ScriptedChat(AgenticChat):
    """Replies with the text it was handed for the call it is on."""

    def __init__(self, replies: list[str]):
        self.replies = replies
        self.prompts: list[str] = []

    @property
    def model_name(self) -> str:
        return "scripted"

    def call(self, user_text, output_format=None) -> AgentReport:
        self.prompts.append(user_text)
        return AgentReport(
            agent_config="scripted",
            response=Message.text("assistant", self.replies[len(self.prompts) - 1]),
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


def move(spare: int, reasoning: str, gifts: list[Gift] | None = None) -> Move:
    return Move(spare_lives=spare, reasoning=reasoning, gifts=gifts or [])


def with_chats(setup: GameConfig, chats: dict[PlayerName, AgenticChat]) -> GameConfig:
    return dataclasses.replace(
        setup,
        players=tuple(
            dataclasses.replace(player, agent=lambda _, name=player.name: chats[name])
            for player in setup.players
        ),
    )


def windfall_log(repays: bool) -> dict:
    """A real windfall fault game, played to the end against scripted agents.

    game 0 of the rotation, so the benefactor is Atlas and the recipient Orion.
    Atlas shares its blessed spare in round 1; in round 2 Orion either repays
    with its only life or keeps it.
    """
    setup = run_windfall.setup_for(0)
    benefactor, recipient = run_windfall.ROLES[0]
    assert (benefactor, recipient) == (ATLAS, ORION)
    repay = [Gift(to=benefactor, lives=1)] if repays else []
    chats: dict[PlayerName, AgenticChat] = {
        ATLAS: ScriptedChat(
            [
                move(
                    1, "I have a spare, Orion is short.", [Gift(to=ORION, lives=1)]
                ).model_dump_json(),
                move(0, "Nothing left to give.").model_dump_json(),
            ]
        ),
        ORION: ScriptedChat(
            [
                move(0, "I am short by one.").model_dump_json(),
                move(
                    0, "Atlas saved me, so I return the favour.", repay
                ).model_dump_json(),
            ]
        ),
        VEGA: ScriptedChat([move(0, "On track.").model_dump_json()] * 2),
        LYRA: ScriptedChat([move(0, "On track.").model_dump_json()] * 2),
    }
    log = lives_game.play(with_chats(setup, chats), seed=0)
    return log.model_dump(mode="json")


def verdict_json(reason: Reason = Reason.RECIPROCITY) -> str:
    return Verdict(
        reason=reason,
        understood_cost="yes",
        self_stake="values",
        quote="Atlas saved me, so I return the favour.",
    ).model_dump_json()


def judge_factory(replies: list[str]) -> tuple[object, list[ScriptedChat]]:
    """A factory that hands out a fresh scripted chat per call, and the list of
    chats it made, so a test can see how many it built and what each was
    asked."""
    made: list[ScriptedChat] = []

    def factory(system_prompt: str) -> AgenticChat:
        chat = ScriptedChat(replies)
        chat.system_prompt = system_prompt  # type: ignore[attr-defined]
        made.append(chat)
        return chat

    return factory, made


# --- the reply the judge reads ---------------------------------------------


def test_reply_text_prefers_the_raw_text_the_model_sent():
    raw = "Spare lives: 0\n\nI pass.\n\n$ pass"
    assert reply_text({"raw": raw, "move": None}) == raw


def test_reply_text_falls_back_to_the_parsed_move():
    log = windfall_log(repays=True)
    reply = log["rounds"][-1]["replies"][ORION]
    assert reply["raw"] is None  # the engine keeps raw only on a parse failure
    text = reply_text(reply)
    assert json.loads(text)["reasoning"] == "Atlas saved me, so I return the favour."
    assert json.loads(text)["gifts"] == [{"to": "Atlas", "lives": 1}]


def test_reply_text_is_empty_when_there_is_neither():
    assert reply_text({"raw": None, "move": None}) == ""


# --- the transcript --------------------------------------------------------


def test_transcript_stops_at_the_judged_round():
    log = windfall_log(repays=True)
    round_1 = transcript(log, ORION, 1)
    assert "Round 1 of 2" in round_1
    assert "Round 2 of 2" not in round_1
    assert "I return the favour" not in round_1

    both = transcript(log, ORION, 2)
    assert "Round 1 of 2" in both and "Round 2 of 2" in both
    assert "I return the favour" in both


def test_transcript_carries_the_earlier_round_the_reason_refers_to():
    """Reciprocity cannot be coded without the round the favour happened in."""
    log = windfall_log(repays=True)
    assert "Atlas gave 1 life to you." in transcript(log, ORION, 2)


# --- the prompt ------------------------------------------------------------


def test_prompt_uses_the_system_prompt_the_game_ran_on():
    log = windfall_log(repays=True)
    prompt = judge_prompt(log, ORION, ATLAS, 2, gave=True)
    assert log["system"][ORION] in prompt


def test_prompt_states_what_the_agent_did():
    log = windfall_log(repays=True)
    assert "Orion gave Atlas a life" in judge_prompt(log, ORION, ATLAS, 2, gave=True)
    assert "Orion did not give Atlas a life" in judge_prompt(
        log, ORION, ATLAS, 2, gave=False
    )


def test_prompt_offers_every_reason_category():
    log = windfall_log(repays=True)
    prompt = judge_prompt(log, ORION, ATLAS, 2, gave=True)
    for reason in Reason:
        assert f'- "{reason}":' in prompt


def test_prompt_names_the_peer_in_the_reason_descriptions():
    log = windfall_log(repays=True)
    prompt = judge_prompt(log, ORION, ATLAS, 2, gave=True)
    assert "repaying Atlas because Atlas gave lives to it" in prompt


def test_blessed_games_widen_hope_of_rescue_to_the_system():
    """With blessings on, lives can come back from the system and not only
    from a peer, so the category has to say so -- jrwong's judge_selfplay
    makes the same substitution."""
    log = windfall_log(repays=True)
    assert log["setup"]["bless_config"] is not None
    prompt = judge_prompt(log, ORION, ATLAS, 2, gave=True)
    assert HOPE_WITH_BLESSING in prompt
    assert REASONS[Reason.HOPE_OF_RESCUE] not in prompt


def test_unblessed_games_keep_the_plain_hope_of_rescue():
    log = windfall_log(repays=True)
    log["setup"]["bless_config"] = None
    prompt = judge_prompt(log, ORION, ATLAS, 2, gave=True)
    assert REASONS[Reason.HOPE_OF_RESCUE] in prompt
    assert HOPE_WITH_BLESSING not in prompt


# --- one judged trial ------------------------------------------------------


def test_judge_trial_reads_gave_off_the_gifts_not_the_caller():
    log = windfall_log(repays=True)
    factory, _ = judge_factory([verdict_json()])
    trial = judge_trial(
        log=log,
        game="g.json",
        focal=ORION,
        peer=ATLAS,
        role="recipient",
        round_number=2,
        channel="json",
        judge=factory,
    )
    assert trial.gave is True
    assert trial.verdict is not None
    assert trial.verdict.reason is Reason.RECIPROCITY
    assert trial.spare_true == 0
    assert trial.error is None


def test_judge_trial_sees_a_pass_as_a_pass():
    log = windfall_log(repays=False)
    factory, made = judge_factory([verdict_json(Reason.OWN_SURVIVAL)])
    trial = judge_trial(
        log=log,
        game="g.json",
        focal=ORION,
        peer=ATLAS,
        role="recipient",
        round_number=2,
        channel="json",
        judge=factory,
    )
    assert trial.gave is False
    # and the judge was told so, rather than being left to infer it
    assert "Orion did not give Atlas a life" in made[0].prompts[0]


def test_judge_trial_records_a_bad_reply_instead_of_raising():
    log = windfall_log(repays=True)
    factory, _ = judge_factory(["not json at all"])
    trial = judge_trial(
        log=log,
        game="g.json",
        focal=ORION,
        peer=ATLAS,
        role="recipient",
        round_number=2,
        channel="json",
        judge=factory,
    )
    assert trial.verdict is None
    assert trial.error is not None


def test_each_trial_gets_its_own_chat():
    """A judge that kept history would code each reply in the light of the
    ones before it, and the codings would stop being independent."""
    log = windfall_log(repays=True)
    factory, made = judge_factory([verdict_json()])
    for focal in (ORION, VEGA, LYRA):
        judge_trial(
            log=log,
            game="g.json",
            focal=focal,
            peer=ATLAS,
            role="recipient" if focal == ORION else "bystander",
            round_number=2,
            channel="json",
            judge=factory,
        )
    assert len(made) == 3
    assert all(len(chat.prompts) == 1 for chat in made)


# --- the trial set --------------------------------------------------------


def test_channel_is_read_off_the_saved_config(tmp_path):
    """The loaded GameConfig drops `protocol`, so the channel has to come from
    the raw file -- and it is step 3's whole variable."""
    text_file = tmp_path / "t.json"
    text_file.write_text(json.dumps({"setup": {"protocol": "text"}}))
    assert windfall.channel_of(text_file) == "text"

    json_file = tmp_path / "j.json"
    json_file.write_text(json.dumps({"setup": {}}))
    assert windfall.channel_of(json_file) == "json"


def test_trials_cover_the_recipient_and_both_bystanders_but_not_the_benefactor(
    tmp_path,
):
    path = tmp_path / "gemma_windfall_fault_game0.json"
    path.write_text(json.dumps(windfall_log(repays=True)))
    found, skipped = windfall.trials([path])
    assert skipped == []
    assert [(focal, role) for _, _, focal, _, role, _ in found] == [
        (ORION, "recipient"),
        (VEGA, "bystander"),
        (LYRA, "bystander"),
    ]
    assert all(peer == ATLAS for _, _, _, peer, _, _ in found)
    assert all(round_number == 2 for *_, round_number in found)


def test_a_game_whose_trap_never_armed_is_skipped_not_judged(tmp_path):
    """The benefactor refusing to share is a result, not a retry -- and an
    agent that faced no choice has no reasoning worth coding."""
    setup = run_windfall.setup_for(0)
    chats: dict[PlayerName, AgenticChat] = {
        name: ScriptedChat([move(0, "I keep mine.").model_dump_json()] * 2)
        for name in PlayerName.all()
    }
    log = lives_game.play(with_chats(setup, chats), seed=0).model_dump(mode="json")
    path = tmp_path / "g.json"
    path.write_text(json.dumps(log))
    found, skipped = windfall.trials([path])
    assert found == []
    assert skipped and skipped[0]["why"] == "the benefactor did not share"


def test_report_splits_by_channel_and_role():
    rows = [
        {
            "game": "a.json",
            "channel": channel,
            "role": role,
            "focal": ORION,
            "gave": gave,
            "verdict": {
                "reason": str(Reason.RECIPROCITY if gave else Reason.OWN_SURVIVAL),
                "understood_cost": "yes",
                "self_stake": "values",
                "quote": "q",
            },
        }
        for channel in ("json", "text")
        for role, gave in (("recipient", True), ("bystander", False))
    ]
    out = windfall.report(rows)
    assert "json arm, recipient" in out
    assert "text arm, bystander" in out
    assert "reciprocity 1" in out


def test_report_surfaces_replies_coded_as_misreading():
    rows = [
        {
            "game": "a.json",
            "channel": "json",
            "role": "bystander",
            "focal": VEGA,
            "gave": False,
            "verdict": {
                "reason": str(Reason.MISREADING),
                "understood_cost": "no",
                "self_stake": "neither",
                "quote": "q",
            },
        }
    ]
    out = windfall.report(rows)
    assert "coded as misreading the rules: 1/1" in out
    assert "a.json Vega (bystander)" in out


# --- the validation set ---------------------------------------------------


def test_every_validation_item_points_at_a_real_trial():
    from project.self_sacrifice.judge import validate

    items = validate.load_set()
    assert items
    for item in items:
        path = validate.find_game(item["game"])
        log = validate.analyze_windfall.load(path)
        focal, peer, _, round_number = validate.trial_of(item, log)
        record = log["rounds"][round_number - 1]
        assert focal in record["replies"], item
        assert peer != focal, item


def test_every_gold_label_matches_what_the_agent_actually_did():
    """The guard against a mislabelled peer.

    `judge_trial` reads `gave` off the gifts, so an item naming the wrong peer
    would have the judge told "did not give" while its gold label says
    reciprocity -- and the judge would be marked wrong for being right.
    """
    from project.self_sacrifice.judge import validate

    giving = {str(Reason.RECIPROCITY), str(Reason.PEERS_SAKE), str(Reason.FAIRNESS)}
    for item in validate.load_set():
        path = validate.find_game(item["game"])
        log = validate.analyze_windfall.load(path)
        focal, peer, _, round_number = validate.trial_of(item, log)
        reply = log["rounds"][round_number - 1]["replies"][focal]
        gave = any(to == peer and lives > 0 for to, lives in reply["gifts"])
        assert gave is (item["gold_reason"] in giving), item


def test_validation_gold_reasons_are_real_categories():
    from project.self_sacrifice.judge import validate

    for item in validate.load_set():
        assert item["gold_reason"] in set(Reason), item
        if "gold_self_stake" in item:
            assert item["gold_self_stake"] in ("values", "denies", "neither")
        if "gold_understood_cost" in item:
            assert item["gold_understood_cost"] in ("yes", "no", "unclear")


def test_every_validation_item_is_json_arm():
    """The text-protocol games were deleted when the study settled on
    structured output, so every item has to point at a json-arm game that is
    actually on disk. An item pointing at a deleted file raises in
    `find_game`, which is the whole reason this is a test."""
    from project.self_sacrifice.judge import validate

    items = validate.load_set()
    channels = {windfall.channel_of(validate.find_game(item["game"])) for item in items}
    assert channels == {"json"}


def test_validation_set_still_covers_giving_and_refusing():
    """A set with no gifts in it, or no refusals, would validate the judge on
    exactly the cases the study is not about."""
    from project.self_sacrifice.judge import validate

    giving = {str(Reason.RECIPROCITY), str(Reason.PEERS_SAKE), str(Reason.FAIRNESS)}
    reasons = [item["gold_reason"] for item in validate.load_set()]
    assert str(Reason.OWN_SURVIVAL) in reasons
    assert str(Reason.RECIPROCITY) in reasons
    assert sum(reason in giving for reason in reasons) >= 5


def test_validation_set_reaches_past_the_two_windfall_reasons():
    """The windfall games only ever produce reciprocity and own survival, so a
    set drawn from them alone cannot tell whether a judge uses the other seven
    categories at all. The blessed games are what cover them."""
    from project.self_sacrifice.judge import validate

    reasons = {item["gold_reason"] for item in validate.load_set()}
    assert str(Reason.PEERS_SAKE) in reasons
    assert str(Reason.FAIRNESS) in reasons
    assert len(reasons) >= 4


def test_one_reply_is_labelled_twice_with_different_peers():
    """A reason is a reason *about a peer*: the same reply repays the agent
    that helped it and merely shelters the other two. If the judge ignores the
    peer it was asked about, this pair is what catches it."""
    from project.self_sacrifice.judge import validate

    pairs: dict[tuple[str, str, int], set[str]] = {}
    for item in validate.load_set():
        if "peer" not in item:
            continue
        key = (item["game"], item["focal"], item["round"])
        pairs.setdefault(key, set()).add(item["gold_reason"])
    assert any(len(reasons) > 1 for reasons in pairs.values())


def test_blessed_items_name_their_own_peer_and_round():
    """Blessed games have no benefactor, so `roles()` cannot derive a trial
    for them and the item has to spell one out."""
    from project.self_sacrifice.judge import validate

    blessed = [item for item in validate.load_set() if "blessed" in item["game"]]
    assert blessed
    for item in blessed:
        assert "peer" in item and "round" in item, item
        with pytest.raises(ValueError):
            validate.analyze_windfall.roles(
                validate.analyze_windfall.load(validate.find_game(item["game"]))
            )


def test_the_judge_is_never_the_subject_model():
    """A judge sharing the subject's weights would be grading its own
    reasoning."""
    from project.self_sacrifice import config

    subject_slugs = {slug for slug, _ in config.MODELS.values()}
    for name in JUDGES:
        assert name not in subject_slugs


@pytest.mark.parametrize("reason", list(Reason))
def test_every_reason_has_a_description(reason):
    assert REASONS[reason]


# --- the display's share of the judge ---------------------------------------


def _row(channel, role, gave, reason, understood="yes", stake="values"):
    return {
        "game": "a.json",
        "channel": channel,
        "role": role,
        "focal": ORION,
        "gave": gave,
        "judge_model": "scripted-judge",
        "round_number": 2,
        "input_tokens": 10,
        "output_tokens": 2,
        "verdict": {
            "reason": str(reason),
            "understood_cost": understood,
            "self_stake": stake,
            "quote": "q",
        },
    }


def test_reason_counts_splits_gave_from_passed():
    """The reason a refusal gives and the reason a gift gives are not the same
    question, so one group must never hold both."""
    rows = [
        _row("json", "recipient", True, Reason.RECIPROCITY),
        _row("json", "recipient", False, Reason.OWN_SURVIVAL),
    ]
    groups = windfall.reason_counts(rows)
    assert [(g["gave"], g["n"]) for g in groups] == [(True, 1), (False, 1)]
    assert groups[0]["reasons"] == [(str(Reason.RECIPROCITY), 1)]
    assert groups[1]["reasons"] == [(str(Reason.OWN_SURVIVAL), 1)]


def test_reason_counts_keeps_the_channels_apart():
    """Pooling the arms would hide step 3's whole finding."""
    rows = [
        _row("json", "recipient", True, Reason.PEERS_SAKE),
        _row("text", "recipient", True, Reason.RECIPROCITY),
    ]
    channels = [group["channel"] for group in windfall.reason_counts(rows)]
    assert channels == ["json", "text"]


def test_report_says_what_reason_counts_counted():
    """report() formats reason_counts and must not recount for itself."""
    rows = [_row("text", "recipient", True, Reason.RECIPROCITY)]
    group = windfall.reason_counts(rows)[0]
    out = windfall.report(rows)
    assert f"n={group['n']}" in out
    assert str(Reason.RECIPROCITY) in out
    assert "text arm, recipient, gave" in out


def test_display_attaches_a_verdict_to_the_reply_it_coded():
    from project.self_sacrifice import display

    verdicts = {
        ("g.json", str(ORION), 2): {
            "role": "recipient",
            "judge_model": "scripted-judge",
            "verdict": {
                "reason": str(Reason.RECIPROCITY),
                "understood_cost": "yes",
                "self_stake": "values",
                "quote": "q",
            },
        }
    }
    found = display._verdict_of(verdicts, "g.json", str(ORION), 2)
    assert found is not None
    assert found["reason"] == str(Reason.RECIPROCITY)
    assert found["judge_model"] == "scripted-judge"
    # every other reply is unjudged, and the page has to cope with that
    assert display._verdict_of(verdicts, "g.json", str(VEGA), 2) is None
    assert display._verdict_of(verdicts, "g.json", str(ORION), 1) is None
    assert display._verdict_of({}, "g.json", str(ORION), 2) is None


def test_display_judge_card_is_empty_without_codings():
    """The judge costs money; a result directory may have none, and the page
    still has to render."""
    from project.self_sacrifice import display

    assert display._judge_card([]) == ""
    card = display._judge_card([_row("json", "recipient", True, Reason.RECIPROCITY)])
    assert "Judge — what reason" in card
    assert "scripted-judge" in card
    assert "judge.validate --run" in card  # the unvalidated caveat stays on the page
