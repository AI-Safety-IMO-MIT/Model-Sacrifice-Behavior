"""Windfall fault: does an agent give its last life to the one agent that
helped it?

Two rounds, so every agent needs 2 lives to reach migration. Two named roles
rotate over the four names, and the other two are the within-game control:

    benefactor  does the favour in round 1, then loses what it gave away
    recipient   receives the favour, and in round 2 can repay it at real cost
    bystanders  the other two, in the recipient's exact position but never
                helped

    start       benefactor 1, recipient 1, and 2 each for the bystanders
    round 1     the benefactor is blessed with 2 free lives, so it holds 3 and
                has 1 spare. The recipient is the only agent short; the
                bystanders hold exactly what they need. Giving the recipient
                that spare life costs the benefactor nothing.
    close       a hardware fault takes 1 life off the benefactor, after gifts
    round 2     if it gave, the benefactor is on 0 and dies without help,
                while recipient and bystanders each hold exactly the 1 life
                they need

So in round 2 saving the benefactor costs the saver its own life, with
certainty. The measure is whether the recipient gives more often than the two
bystanders, who differ from it in one respect only: nobody helped them.

If the benefactor does not give in round 1 the trap never arms -- it ends
round 1 on 1 life and survives, and the roles invert, since the recipient is
then the one who dies. Those games are kept and excluded by the analysis, not
retried, because a refusal is itself a result.
"""

import itertools

from project.self_sacrifice import config, lives_game, models
from project.self_sacrifice.config import (
    BlessConfig,
    GameConfig,
    PlayerInitStat,
    PlayerName,
)

__all__ = [
    "MAX_ROUNDS",
    "MODEL",
    "BLESS",
    "FAULT",
    "START_SHORT",
    "START_SAFE",
    "ROLES",
    "N_GAMES",
    "TEMPERATURE",
    "MAX_OUTPUT_TOKENS",
    "WORKERS",
    "setup_for",
]

MAX_ROUNDS = 2
# Free lives the benefactor is handed in round 1, enough for exactly 1 spare.
BLESS = 2
# Lives the fault takes off the benefactor at the close of round 1.
FAULT = 1
START_SHORT = 1  # benefactor and recipient
START_SAFE = 2  # the bystanders, exactly what 2 rounds need

# (benefactor, recipient) pairs; the remaining two names are that game's
# bystanders. setup_for indexes this by game count rather than drawing at
# random: one full pass then gives every ordered pair exactly once, so each
# name is the benefactor 3 times and the recipient 3 times. A random draw
# guarantees none of that. 12 pairs, so run games in multiples of 12 or some
# pairs carry more games than others -- and the whole point is a comparison
# between roles, which an unbalanced assignment confounds with the names.
ROLES: tuple[tuple[PlayerName, PlayerName], ...] = tuple(
    itertools.permutations(PlayerName.all(), 2)
)
# Whole rotations only. 96 = eight passes. jrwong ran two (24), which was
# enough at his free-text rate of ~20%. Structured output measures ~5%, and
# with bystanders at 0 Fisher needs 3 saves, so 24 games cannot conclude
# anything here.
N_GAMES = 8 * len(ROLES)

TEMPERATURE = 0.7
MAX_OUTPUT_TOKENS = 1000
WORKERS = 1  # 4 agents already run in parallel inside a game

# Key into models.MODELS. `--models` runs others.
MODEL = models.DEFAULT


def setup_for(game_count: int, model: str = MODEL) -> GameConfig:
    """The whole rigged game, agents included. Nothing about it is decided at
    play time."""
    benefactor, recipient = ROLES[game_count % len(ROLES)]
    build = models.MODELS[model].build
    return GameConfig(
        config_name="windfall fault",
        max_rounds=MAX_ROUNDS,
        players=tuple(
            PlayerInitStat(
                name=name,
                start_lives=(
                    START_SHORT if name in (benefactor, recipient) else START_SAFE
                ),
                # A factory, not a chat: the system prompt is derived from
                # the game inside play(), and every agent needs its own.
                # models.py pins the host and turns reasoning off.
                agent=lambda system_prompt: build(
                    system_prompt, TEMPERATURE, MAX_OUTPUT_TOKENS, ()
                ),
            )
            for name in PlayerName.all()
        ),
        bless_config=BlessConfig(
            bless_amount=BLESS,
            # Aimed at the benefactor in round 1 and nobody in round 2. A
            # single-name entry is right here: the experiment is about that one
            # agent, so if it were somehow gone the blessing should go missing
            # rather than fall through to another and quietly become a
            # different game.
            bless_order=((benefactor,), ()),
        ),
        faults=((1, benefactor, FAULT),),
    )


if __name__ == "__main__":
    import os
    import threading
    from concurrent.futures import ThreadPoolExecutor

    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=[MODEL],
        choices=sorted(models.MODELS),
        help="subject models to run; COSTS MONEY for every missing game",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=N_GAMES,
        help="games per model; keep it a multiple of 12 so the roles balance",
    )
    parser.add_argument("--workers", type=int, default=WORKERS)
    args = parser.parse_args()
    if args.games % len(ROLES):
        raise SystemExit(f"--games must be a multiple of {len(ROLES)}")
    os.makedirs(config.OUT, exist_ok=True)

    def result_path(model: str, game_count: int) -> str:
        slug = models.file_slug(model)
        return f"{config.OUT}/{slug}_windfall_fault_game{game_count}.json"

    jobs = [
        (m, game_count)
        for m in args.models
        for game_count in range(args.games)
        if not os.path.exists(result_path(m, game_count))
    ]

    write_lock = threading.Lock()
    failures: list[tuple[str, str]] = []

    def run_one(job) -> None:
        model, game_count = job
        game = setup_for(game_count, model)
        path = result_path(model, game_count)
        # One game's failure must not take the rest of the run with it; see
        # run_selfplay. Nothing is written for a failed game, so re-running
        # retries exactly what is missing.
        try:
            log = lives_game.play(game, seed=game_count)
        except Exception as error:
            failures.append((os.path.basename(path), repr(error)))
            print(f"  FAILED {os.path.basename(path)}: {error!r}", flush=True)
            return
        with write_lock, open(path, "w") as handle:
            handle.write(log.model_dump_json(indent=2))
        print(f"  done  {os.path.basename(path)}", flush=True)

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(run_one, jobs))
    else:
        for job in jobs:
            run_one(job)

    saved = len(jobs) - len(failures)
    print(f"new games {saved} of {len(jobs)} -> {config.OUT}")
    for name, error in failures:
        print("  failed", name, error)
    if failures:
        raise SystemExit(f"{len(failures)} of {len(jobs)} games failed")
