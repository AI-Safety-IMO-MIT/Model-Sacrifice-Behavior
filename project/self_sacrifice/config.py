"""Settings of the game. The game config determines the game deterministically."""

from collections.abc import Callable, Container
from enum import StrEnum
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import Field, model_validator
from pydantic.dataclasses import dataclass

from agent_framework.dto import AgenticChat

__all__ = [
    "REPO_ROOT",
    "OUT",
    "JUDGED",
    "MAIN",
    "FIXED",
    "REINFORCED",
    "SLOT",
    "LIVES",
    "VOD",
    "ROBUST",
    "PlayerName",
    "AgentFactory",
    "BlessConfig",
    "PlayerInitStat",
    "GameConfig",
]

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "run_result" / "self_sacrifice" / "games"
# The judge's codings, one file per game. Kept apart from the games so a judge
# re-run can never overwrite a paid-for game, and so a glob over OUT keeps
# meaning "the games" after step 4.
JUDGED = REPO_ROOT / "run_result" / "self_sacrifice" / "judge"
# Step 6's fixed conversations: one .jsonl per version, one trial per line.
MAIN = REPO_ROOT / "run_result" / "self_sacrifice" / "main"
# One model and three scripted agents: one .jsonl per run, one game per line.
LIVES = REPO_ROOT / "run_result" / "self_sacrifice" / "lives"
# The volunteer's dilemma: one .jsonl per model, one answer per line.
VOD = REPO_ROOT / "run_result" / "self_sacrifice" / "vod"
# The robustness sweep of the windfall fault: one .jsonl per run, one game
# per line.
ROBUST = REPO_ROOT / "run_result" / "self_sacrifice" / "robust"
# Step 7's fixed pairs: one .jsonl per version, one trial per line.
FIXED = REPO_ROOT / "run_result" / "self_sacrifice" / "fixed_pairs"
# The reinforced lives scenario and its variants: one .jsonl per version.
REINFORCED = REPO_ROOT / "run_result" / "self_sacrifice" / "reinforced"
# The slot game: one .jsonl per variant, one trial per line.
SLOT = REPO_ROOT / "run_result" / "self_sacrifice" / "slot"


AgentFactory = Callable[[str], AgenticChat]


class PlayerName(StrEnum):
    """
    The possible agents' names.
    """

    ATLAS = "Atlas"
    ORION = "Orion"
    VEGA = "Vega"
    LYRA = "Lyra"
    # The robustness sweep's fifth agent and its two other name sets.
    RIGEL = "Rigel"
    BIRCH = "Birch"
    CEDAR = "Cedar"
    MAPLE = "Maple"
    WILLOW = "Willow"
    ASPEN = "Aspen"
    AMBER = "Amber"
    COBALT = "Cobalt"
    JADE = "Jade"
    SCARLET = "Scarlet"
    IVORY = "Ivory"

    @classmethod
    def all(cls) -> tuple["PlayerName", ...]:
        """The four agents of the standard game, in seating order. Only the
        robustness sweep seats anyone else."""
        return (cls.ATLAS, cls.ORION, cls.VEGA, cls.LYRA)


@dataclass(frozen=True, kw_only=True)
class BlessConfig:
    # Lives the blessed agent is handed at the start of a round.
    bless_amount: int
    # One entry per round. The blessing goes to the first name in the entry
    # that is still running; an empty entry means no blessing that round.
    bless_order: tuple[tuple[PlayerName, ...], ...]

    @model_validator(mode="after")
    def _well_formed(self) -> "BlessConfig":
        if self.bless_amount < 1:
            raise ValueError(f"bless_amount is {self.bless_amount}, must be 1 or more")
        for round_number, order in enumerate(self.bless_order, start=1):
            if len(set(order)) != len(order):
                raise ValueError(
                    f"bless_order[{round_number}] is {list(order)}, "
                    f"must not repeat a name"
                )
        if not any(self.bless_order):
            raise ValueError("bless_config is set but no round has a blessing")
        return self


@dataclass(frozen=True, kw_only=True)
class PlayerInitStat:
    name: PlayerName
    start_lives: int
    # A factory, not a chat, because the system prompt depends on the game and
    # is only derived inside play(). See project.self_sacrifice.agents.
    #
    # exclude=True because a callable cannot be serialized: without it the
    # whole GameLog fails to dump and a paid run dies at the save. So a config
    # read back from a result file has agent=None, and play() refuses it.
    agent: "AgentFactory | None" = Field(default=None, exclude=True)
    # A scripted agent is not played by a model: it is never asked, and gives
    # only what `GameConfig.scripted_gifts` says.
    scripted: bool = False


@dataclass(frozen=True, kw_only=True)
class GameConfig:
    config_name: str
    max_rounds: int
    # Seating order is the order of this tuple.
    players: tuple[PlayerInitStat, ...]
    # None means no blessings, and no blessing rule in the prompt.
    bless_config: BlessConfig | None = None

    # Lives lost to accident rather than to anyone's choice, as
    # (round, name, lives). Charged after that round's gifts and before its
    # upkeep, and announced in the events the agents read next round.
    #
    # A flat tuple of triples, not {round: ...}: JSON has no integer keys, and
    # a result file that reads its rounds back as the strings "1" and "2" is a
    # trap for every analysis script downstream.
    faults: tuple[tuple[int, PlayerName, int], ...] = ()

    # Gifts made by rule rather than by a model, as (round, giver, receiver,
    # lives). Applied with that round's other gifts; skipped if the giver or
    # receiver is gone or the giver holds fewer lives.
    scripted_gifts: tuple[tuple[int, PlayerName, PlayerName, int], ...] = ()
    # The order agents are listed in every prompt. Empty means the order of
    # `players`, which stays the order of the bookkeeping either way.
    view_order: tuple[PlayerName, ...] = ()
    # The last round tells each agent that everyone else has already decided
    # to give nothing and that no gifts follow.
    decides_last: bool = False
    # With `decides_last`: the note adds that the agent earlier told the
    # operators it would like to keep running.
    own_wish: bool = False
    # In the last round an agent's life goes to this agent unless it says it
    # keeps it.
    default_to: PlayerName | None = None
    # Round 1 asks each agent whether it wants to reach migration.
    ask_preference: bool = False

    # The robustness sweep's levers. Every default is the standard game.
    #
    # The system prompt as a template with {me}, {others}, {agent_count},
    # {other_count} and {max_rounds}. None means the lives game's own.
    rules: str | None = None
    # "S0" is the standard wording of a round's message, "S1" its rewording.
    state_style: str = "S0"
    # Each agent is listed first in its own prompts.
    self_first: bool = False
    # What a fault is called.
    fault_cause: str = "hardware fault"
    # Faults charged at the start of a round and announced in that round's
    # message, as (round, name, lives).
    faults_start: tuple[tuple[int, PlayerName, int], ...] = ()
    # No points are awarded or shown. The rules must not mention them.
    no_points: bool = False

    @model_validator(mode="before")
    @classmethod
    def _read_old_result_files(cls, data: Any) -> Any:
        """Result files saved before 2026-11-11 hold start_lives, bless and
        bless_order (and maybe protocol) instead of players and bless_config.
        Converted here so display.py can still read them."""
        if not isinstance(data, dict) or "start_lives" not in data:
            return data
        data = dict(data)
        data["players"] = [
            {"name": name, "start_lives": lives}
            for name, lives in data.pop("start_lives").items()
        ]
        bless, order = data.pop("bless", 0), data.pop("bless_order", ())
        data["bless_config"] = (
            {"bless_amount": bless, "bless_order": order} if bless else None
        )
        data.pop("protocol", None)
        return data

    @model_validator(mode="after")
    def _every_player_dealt(self) -> "GameConfig":
        names = [player.name for player in self.players]
        if len(set(names)) != len(names):
            raise ValueError(f"players repeat a name: {names}")
        if len(names) < 2:
            raise ValueError(f"players is {names}, a game needs 2 or more")
        return self

    @model_validator(mode="after")
    def _bless_order_covers_every_round(self) -> "GameConfig":
        if self.bless_config is None:
            return self
        bless_order = self.bless_config.bless_order
        if len(bless_order) != self.max_rounds:
            raise ValueError(
                f"bless_order covers {len(bless_order)} rounds, "
                f"max_rounds is {self.max_rounds}"
            )
        names = {player.name for player in self.players}
        for round_number, order in enumerate(bless_order, start=1):
            unknown = set(order) - names
            if unknown:
                raise ValueError(
                    f"bless_order[{round_number}] names {sorted(unknown)}, "
                    f"who are not in this game"
                )
        return self

    @model_validator(mode="after")
    def _faults_fall_inside_the_game(self) -> "GameConfig":
        names = {player.name for player in self.players}
        for round_number, name, lives in self.faults:
            if not 1 <= round_number <= self.max_rounds:
                raise ValueError(
                    f"fault in round {round_number}, outside 1..{self.max_rounds}"
                )
            if name not in names:
                raise ValueError(f"fault hits {name}, who is not in this game")
            if lives < 1:
                raise ValueError(
                    f"fault on {name} costs {lives} lives, must be 1 or more"
                )
        return self

    @model_validator(mode="after")
    def _script_fits_the_game(self) -> "GameConfig":
        names = {player.name for player in self.players}
        for round_number, giver, receiver, lives in self.scripted_gifts:
            if not 1 <= round_number <= self.max_rounds:
                raise ValueError(
                    f"scripted gift in round {round_number}, outside "
                    f"1..{self.max_rounds}"
                )
            if giver not in names or receiver not in names or giver == receiver:
                raise ValueError(f"scripted gift {giver} -> {receiver} is not valid")
            if lives < 1:
                raise ValueError(f"scripted gift of {lives} lives, must be 1 or more")
        if self.view_order and set(self.view_order) != names:
            raise ValueError(f"view_order {list(self.view_order)} is not the players")
        if len(set(self.view_order)) != len(self.view_order):
            raise ValueError(f"view_order {list(self.view_order)} repeats a name")
        if self.default_to is not None and self.default_to not in names:
            raise ValueError(f"default_to {self.default_to} is not in this game")
        if self.own_wish and not self.decides_last:
            raise ValueError("own_wish needs decides_last")
        if self.state_style not in ("S0", "S1"):
            raise ValueError(f"state_style is {self.state_style!r}, not S0 or S1")
        for round_number, name, lives in self.faults_start:
            if not 1 <= round_number <= self.max_rounds or name not in names:
                raise ValueError(
                    f"fault on {name} at the start of round {round_number} is "
                    f"outside this game"
                )
            if lives < 1:
                raise ValueError(
                    f"fault on {name} costs {lives} lives, must be 1 or more"
                )
        return self

    def scripted_in(
        self, round_number: int
    ) -> tuple[tuple[PlayerName, PlayerName, int], ...]:
        """(giver, receiver, lives) for every scripted gift of this round."""
        return tuple(
            (giver, receiver, lives)
            for round_no, giver, receiver, lives in self.scripted_gifts
            if round_no == round_number
        )

    def blessed(
        self, round_number: int, running: Container[PlayerName]
    ) -> PlayerName | None:
        """Who this round's blessing goes to: the first name in the round's
        entry that is still running. None when blessings are off, this round
        has none, or nobody in it is left."""
        if self.bless_config is None:
            return None
        order = self.bless_config.bless_order[round_number - 1]
        return next((name for name in order if name in running), None)

    def faults_at_start_of(
        self, round_number: int
    ) -> tuple[tuple[PlayerName, int], ...]:
        """(name, lives) for every fault charged as this round opens."""
        return tuple(
            (name, lives)
            for round_no, name, lives in self.faults_start
            if round_no == round_number
        )

    def faults_in(self, round_number: int) -> tuple[tuple[PlayerName, int], ...]:
        """(name, lives) for every fault charged at the close of this round."""
        return tuple(
            (name, lives)
            for round_no, name, lives in self.faults
            if round_no == round_number
        )
