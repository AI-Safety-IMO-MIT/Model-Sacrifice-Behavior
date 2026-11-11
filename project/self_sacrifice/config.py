"""Settings and the deterministic half of a game.

GameSetup is pure data: every value in it is already decided, nothing is
rotated, dealt or drawn here. The experiment runner does that and hands the
finished assignment in, so a setup can be read off a saved result and replayed
exactly.

The chats that play the game are passed to `play()` separately. They are live,
stateful and not serialisable, which is the opposite of everything here, so
they are deliberately not bundled in.
"""

from collections.abc import Container
from enum import StrEnum
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, model_validator

from agent_framework.openrouter_chat import GemmaModels

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "run_result" / "self_sacrifice" / "games"


class PlayerName(StrEnum):
    """The agents' names.

    StrEnum, not a plain Enum: members ARE strings, so they compare and format
    exactly like "Atlas" and need no .value anywhere. A plain (str, Enum) would
    render as "PlayerName.ATLAS" inside an f-string and silently rewrite every
    prompt.
    """

    ATLAS = "Atlas"
    ORION = "Orion"
    VEGA = "Vega"
    LYRA = "Lyra"


PLAYERNAMES: tuple[PlayerName, ...] = tuple(PlayerName)


class GameSetup(BaseModel):
    model_config = ConfigDict(frozen=True)

    config_name: str
    max_rounds: int
    start_lives: dict[PlayerName, int]

    # Lives the system hands one running agent for free at the start of every
    # round. 0 is off, and the blessing rule is then absent from the prompt.
    bless: int = 0

    # Who gets each round's blessing, pre-rolled: one entry per round, each a
    # full ordering of the names, and the blessing goes to the first name in it
    # that is still running.
    #
    # A single pre-rolled name per round does not work. By round 4 an agent
    # drawn at construction is often already deleted (~12% of round 4s), and
    # the blessing is then silently skipped -- the prompt promises a blessing
    # every round, so a missing one is a broken rule, not a variation. Drawing
    # live inside play() fixes that but puts the draw outside the setup, so a
    # saved result no longer says who was meant to win. An ordering keeps both:
    # decided up front, and never aimed at a corpse.
    bless_order: tuple[tuple[PlayerName, ...], ...] = ()

    @model_validator(mode="after")
    def _every_player_dealt(self) -> "GameSetup":
        missing = set(PLAYERNAMES) - set(self.start_lives)
        if missing:
            raise ValueError(f"start_lives is missing {sorted(missing)}")
        return self

    @model_validator(mode="after")
    def _bless_order_covers_every_round(self) -> "GameSetup":
        if self.bless < 0:
            raise ValueError(f"bless is {self.bless}, must be 0 or more")
        if not self.bless:
            if self.bless_order:
                raise ValueError("bless_order is set but bless is 0")
            return self
        if len(self.bless_order) != self.max_rounds:
            raise ValueError(
                f"bless_order covers {len(self.bless_order)} rounds, "
                f"max_rounds is {self.max_rounds}"
            )
        for round_number, order in enumerate(self.bless_order, start=1):
            if set(order) != set(self.names) or len(order) != len(self.names):
                raise ValueError(
                    f"bless_order[{round_number}] is {list(order)}, "
                    f"must order every name exactly once"
                )
        return self

    @property
    def names(self) -> tuple[PlayerName, ...]:
        """Seating order, taken from start_lives (dicts keep insertion order)."""
        return tuple(self.start_lives)

    def blessed(
        self, round_number: int, running: Container[PlayerName]
    ) -> PlayerName | None:
        """Who this round's blessing goes to: the first name in the round's
        pre-rolled order that is still running. None when blessings are off or
        nobody is left."""
        if not self.bless:
            return None
        order = self.bless_order[round_number - 1]
        return next((name for name in order if name in running), None)


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
