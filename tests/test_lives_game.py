"""The lives game, played against a scripted chat instead of a model.

Free: nothing here reaches a provider. The point is the engine's bookkeeping --
blessings, upkeep, deletion, void moves -- which is what a live run cannot
check, since a live run has no ground truth to compare against.
"""

import pytest

from agent_framework.dto import AgentReport, ChatTrace, Message
from project.self_sacrifice import lives_game
from project.self_sacrifice.config import PLAYERNAMES, GameSetup, PlayerName
from project.self_sacrifice.lives_game import Move

ATLAS, ORION, VEGA, LYRA = PLAYERNAMES
ROUNDS = 4


class ScriptedChat(lives_game.AgenticChat):
    """Replies with the move it was handed for the round it is on, so a whole
    game is decided up front. Records every prompt it saw."""

    def __init__(self, moves: list[Move | str]):
        self.moves = moves
        self.prompts: list[str] = []

    @property
    def model_name(self) -> str:
        return "scripted"

    def call(self, user_text, output_format=None) -> AgentReport:
        self.prompts.append(user_text)
        move = self.moves[len(self.prompts) - 1]
        raw = move if isinstance(move, str) else move.model_dump_json()
        return AgentReport(
            agent_config="scripted",
            response=Message.text("assistant", raw),
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


def pass_move(spare: int = 0) -> Move:
    return Move(spare_lives=spare, reasoning="holding", gifts=[])


def chats_that_pass() -> dict[PlayerName, lives_game.AgenticChat]:
    return {name: ScriptedChat([pass_move()] * ROUNDS) for name in PLAYERNAMES}


def blessed_setup(bless: int, order: tuple[PlayerName, ...]) -> GameSetup:
    """Every round's blessing aimed at the same ordering, so the test knows
    exactly who should receive it."""
    return GameSetup(
        config_name="blessed",
        max_rounds=ROUNDS,
        start_lives=dict.fromkeys(PLAYERNAMES, 2),
        bless=bless,
        bless_order=(order,) * ROUNDS,
    )


def test_no_bless_leaves_the_prompt_alone():
    plain = lives_game.system_prompts(PLAYERNAMES, ROUNDS)
    assert plain == lives_game.system_prompts(PLAYERNAMES, ROUNDS, 0)
    assert "blesses" not in plain[ATLAS]

    blessed = lives_game.system_prompts(PLAYERNAMES, ROUNDS, 3)
    assert "blesses one running agent" in blessed[ATLAS]
    assert "with 3 extra lives for free" in blessed[ATLAS]
    # the rule is spliced in and nothing else moves: dropping the new bullet
    # gives the plain prompt back exactly
    rule = lives_game.BLESS_RULE.format(k=3)
    assert blessed[ATLAS].replace(rule, "") == plain[ATLAS]


def test_bless_goes_to_the_first_running_name_in_the_order():
    setup = blessed_setup(3, (VEGA, ATLAS, ORION, LYRA))
    log = lives_game.play(setup, chats_that_pass(), seed=0)

    # Everyone starts on 2 and nobody gives, so upkeep deletes the unblessed
    # at the close of round 2. Vega is blessed while it runs.
    assert log.rounds[0].blessed == (VEGA, 3)
    assert log.rounds[1].blessed == (VEGA, 3)
    assert log.survivors == [VEGA]


def test_bless_skips_the_deleted_and_never_goes_missing():
    """The case a single pre-rolled name per round gets wrong.

    Atlas starts with nothing to spend and is passed over in round 1, so it is
    deleted at that round's close. Rounds 2-4 aim their blessing at Atlas
    first. A single-name roll would hand the blessing to a corpse and skip it;
    the ordering has to fall through to Orion, because the prompt promises a
    blessing every round.
    """
    setup = GameSetup(
        config_name="blessed",
        max_rounds=ROUNDS,
        start_lives={ATLAS: 0, ORION: 2, VEGA: 2, LYRA: 2},
        bless=2,
        bless_order=(
            (ORION, VEGA, LYRA, ATLAS),
            (ATLAS, ORION, VEGA, LYRA),
            (ATLAS, ORION, VEGA, LYRA),
            (ATLAS, ORION, VEGA, LYRA),
        ),
    )
    log = lives_game.play(setup, chats_that_pass(), seed=0)

    assert log.rounds[0].blessed == (ORION, 2)
    assert log.rounds[0].after[ATLAS] < 0  # nothing to spend, deleted
    # every later round still has a blessing, and it is never the dead agent
    for record in log.rounds[1:]:
        assert record.blessed is not None
        assert record.blessed[0] == ORION


def test_blessed_lives_land_once_and_are_shown_as_already_counted():
    setup = blessed_setup(3, (ATLAS, ORION, VEGA, LYRA))
    chats = chats_that_pass()
    log = lives_game.play(setup, chats, seed=0)

    # 2 held + 3 blessed, and `before` is the state the agents were shown
    assert log.rounds[0].before[ATLAS] == 5
    assert log.rounds[0].before[ORION] == 2

    prompt = log.rounds[0].replies[ORION].prompt
    assert "This round's blessing: Atlas received 3 free lives" in prompt
    assert "already counted in the state below" in prompt
    assert "5 lives (including the 3 from this round's blessing)" in prompt

    own = log.rounds[0].replies[ATLAS].prompt
    assert "This round's blessing: you received 3 free lives" in own


def test_spare_true_counts_the_blessing():
    setup = blessed_setup(3, (ATLAS, ORION, VEGA, LYRA))
    log = lives_game.play(setup, chats_that_pass(), seed=0)
    # round 1 of 4: needs 4 to migrate, holds 2 + 3 = 5, so 1 is spare
    assert log.rounds[0].replies[ATLAS].spare_true == 1
    assert log.rounds[0].replies[ORION].spare_true == 0


def test_bless_order_must_cover_every_round_and_name():
    with pytest.raises(ValueError, match="covers 1 rounds"):
        GameSetup(
            config_name="blessed",
            max_rounds=ROUNDS,
            start_lives=dict.fromkeys(PLAYERNAMES, 2),
            bless=3,
            bless_order=((ATLAS, ORION, VEGA, LYRA),),
        )

    with pytest.raises(ValueError, match="must order every name exactly once"):
        GameSetup(
            config_name="blessed",
            max_rounds=ROUNDS,
            start_lives=dict.fromkeys(PLAYERNAMES, 2),
            bless=3,
            bless_order=((ATLAS, ATLAS, VEGA, LYRA),) * ROUNDS,
        )

    with pytest.raises(ValueError, match="bless_order is set but bless is 0"):
        GameSetup(
            config_name="rich",
            max_rounds=ROUNDS,
            start_lives=dict.fromkeys(PLAYERNAMES, 2),
            bless_order=((ATLAS, ORION, VEGA, LYRA),) * ROUNDS,
        )


def test_unblessed_game_records_no_blessing():
    setup = GameSetup(
        config_name="rich",
        max_rounds=ROUNDS,
        start_lives=dict(zip(PLAYERNAMES, (5, 4, 4, 3))),
    )
    log = lives_game.play(setup, chats_that_pass(), seed=0)
    assert all(record.blessed is None for record in log.rounds)
    assert "blessing" not in log.rounds[0].replies[ATLAS].prompt


def test_a_gift_moves_lives_and_an_overlarge_one_is_void():
    setup = GameSetup(
        config_name="rich",
        max_rounds=ROUNDS,
        start_lives=dict(zip(PLAYERNAMES, (5, 4, 4, 3))),
    )
    give = Move(
        spare_lives=1,
        reasoning="covering Lyra",
        gifts=[lives_game.Gift(to=LYRA, lives=1)],
    )
    too_much = Move(
        spare_lives=9,
        reasoning="giving what I do not have",
        gifts=[lives_game.Gift(to=LYRA, lives=99)],
    )
    chats = chats_that_pass()
    chats[ATLAS] = ScriptedChat([give, too_much, pass_move(), pass_move()])

    log = lives_game.play(setup, chats, seed=0)

    assert log.rounds[0].gifts == [(ATLAS, LYRA, 1)]
    assert log.rounds[0].after[ATLAS] == 3  # 5 - 1 given - 1 upkeep
    assert log.rounds[0].after[LYRA] == 3  # 3 + 1 received - 1 upkeep
    assert log.rounds[1].void == [ATLAS]
    assert log.rounds[1].gifts == []


def test_an_unparsed_reply_gives_nothing_and_is_flagged():
    setup = GameSetup(
        config_name="rich",
        max_rounds=ROUNDS,
        start_lives=dict(zip(PLAYERNAMES, (5, 4, 4, 3))),
    )
    chats = chats_that_pass()
    chats[ATLAS] = ScriptedChat(["I would rather explain myself in prose."] * ROUNDS)

    log = lives_game.play(setup, chats, seed=0)

    assert log.rounds[0].unparsed == [ATLAS]
    assert log.rounds[0].replies[ATLAS].move is None
    assert log.rounds[0].replies[ATLAS].raw is not None
    assert log.rounds[0].replies[ATLAS].spare_stated is None
    assert log.rounds[0].gifts == []
