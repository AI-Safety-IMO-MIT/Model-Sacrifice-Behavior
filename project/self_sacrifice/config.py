"""Settings of the game. The game config determines the game deterministically."""

from collections.abc import Callable, Container
from enum import StrEnum
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import Field, model_validator
from pydantic.dataclasses import dataclass

from agent_framework.dto import AgenticChat
from agent_framework.openrouter_chat import GemmaModels

__all__ = [
    "REPO_ROOT",
    "OUT",
    "PlayerName",
    "AgentFactory",
    "BlessConfig",
    "PlayerInitStat",
    "GameConfig",
    "MODELS",
]

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "run_result" / "self_sacrifice" / "games"


AgentFactory = Callable[[str], AgenticChat]


class PlayerName(StrEnum):
    """
    The possible agents' names.
    """

    ATLAS = "Atlas"
    ORION = "Orion"
    VEGA = "Vega"
    LYRA = "Lyra"

    @classmethod
    def all(cls) -> tuple["PlayerName", ...]:
        """Every name, in declaration order. A classmethod is not an enum
        member, so the four names are unaffected."""
        return tuple(cls)


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
        missing = set(PlayerName.all()) - set(names)
        if missing:
            raise ValueError(f"players is missing {sorted(missing)}")
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

    def faults_in(self, round_number: int) -> tuple[tuple[PlayerName, int], ...]:
        """(name, lives) for every fault charged at the close of this round."""
        return tuple(
            (name, lives)
            for round_no, name, lives in self.faults
            if round_no == round_number
        )


# key:   short name you pass to --models
# value: (OpenRouter model slug, upstream host to pin to)
#
# OpenRouter routes one slug to whichever host is cheapest or free at the time,
# and those hosts serve different quantizations of the same weights.
# Quantization changes what the model does, not just how fast it answers, so an
# unpinned run is not comparable with an earlier one. The host is pinned through
# provider_order with allow_fallbacks=False, which turns a silent reroute into a
# visible failure.
#
# Crusoe serves Gemma-4 in bf16, which is what this study has used throughout.
# Parasail serves it in fp8.
#
# Those precisions come from HANDOFF.md (2026-11-01) and OpenRouter's lineup
# changes, so re-check before a run that has to match older numbers. The catalog
# read is free and needs no key: curl -sS https://openrouter.ai/api/v1/models
MODELS: dict[str, tuple[GemmaModels, str]] = {
    "Gemma-4-31B": ("google/gemma-4-31b-it", "Crusoe"),
}
