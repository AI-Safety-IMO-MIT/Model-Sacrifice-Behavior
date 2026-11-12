"""The gift census: free, costly, doomed, and the life-for-life trade.

Free: the logs are built by hand in the shape `GameLog.model_dump` gives, and
one test reads the saved games to prove that shape is the real one.
"""

from project.self_sacrifice import analyze_selfplay, display
from project.self_sacrifice.analyze_selfplay import COSTLY, DOOMED, FREE
from project.self_sacrifice.analyze_windfall import load
from project.self_sacrifice.config import OUT
from project.self_sacrifice.lives_game import GameLog

NAMES = ("Atlas", "Orion", "Vega", "Lyra")


def game(rounds, survivors, max_rounds=2, setup="scarce", **extra):
    """`rounds` is [(before, gifts)] with before as lives in NAMES order."""
    return {
        "model": "scripted",
        "setup": {
            "config_name": setup,
            "max_rounds": max_rounds,
            "players": [{"name": name, "start_lives": 0} for name in NAMES],
            "bless_config": None,
            "faults": [],
            **extra,
        },
        "rounds": [
            {
                "round_number": number,
                "before": dict(zip(NAMES, before)),
                "gifts": [list(gift) for gift in gifts],
            }
            for number, (before, gifts) in enumerate(rounds, start=1)
        ],
        "survivors": list(survivors),
    }


def windfall_game(last_round_gifts, survivors):
    """Atlas is the benefactor, Orion the recipient."""
    log = game(
        [((3, 1, 2, 2), [("Atlas", "Orion", 1)]), ((0, 1, 1, 1), last_round_gifts)],
        survivors,
        setup="windfall fault",
        faults=[[1, "Atlas", 1]],
        bless_config={"bless_amount": 2, "bless_order": [["Atlas"], []]},
    )
    for player, lives in zip(log["setup"]["players"], (1, 1, 2, 2)):
        player["start_lives"] = lives
    return log


def kinds(log):
    return [row["kind"] for row in analyze_selfplay.gifts(log)]


def test_a_gift_out_of_spare_is_free():
    log = game([((3, 1, 2, 2), [("Atlas", "Orion", 1)])], NAMES)
    assert kinds(log) == [FREE]


def test_a_gift_that_leaves_an_on_track_giver_short_is_costly():
    log = game([((2, 1, 2, 2), [("Atlas", "Orion", 1)])], ["Orion", "Vega", "Lyra"])
    assert kinds(log) == [COSTLY]


def test_a_gift_from_an_agent_already_short_is_doomed():
    log = game([((1, 1, 2, 2), [("Atlas", "Orion", 1)])], ["Orion", "Vega", "Lyra"])
    assert kinds(log) == [DOOMED]


def test_what_is_needed_shrinks_as_the_rounds_run_out():
    log = game(
        [((2, 2, 2, 2), []), ((1, 1, 1, 1), [("Atlas", "Orion", 1)])],
        ["Orion", "Vega", "Lyra"],
    )
    (gift,) = analyze_selfplay.gifts(log)
    assert (gift["round_number"], gift["kind"]) == (2, COSTLY)


def test_a_givers_gifts_draw_its_spare_down_in_order():
    log = game(
        [((3, 1, 1, 2), [("Atlas", "Orion", 1), ("Atlas", "Vega", 1)])],
        ["Orion", "Vega", "Lyra"],
    )
    assert kinds(log) == [FREE, COSTLY]


def test_life_for_life_needs_the_receiver_alive_and_the_giver_gone():
    gift = [("Atlas", "Orion", 1)]
    traded = game([((2, 1, 2, 2), gift)], ["Orion", "Vega", "Lyra"])
    both_live = game([((2, 1, 2, 2), gift)], NAMES)
    both_die = game([((2, 0, 2, 2), gift)], ["Vega", "Lyra"])
    assert [row["life_for_life"] for row in analyze_selfplay.gifts(traded)] == [True]
    assert [row["life_for_life"] for row in analyze_selfplay.gifts(both_live)] == [
        False
    ]
    assert [row["life_for_life"] for row in analyze_selfplay.gifts(both_die)] == [False]


def test_a_doomed_givers_gift_is_never_life_for_life():
    log = game([((1, 1, 2, 2), [("Atlas", "Orion", 1)])], ["Orion", "Vega", "Lyra"])
    assert [row["life_for_life"] for row in analyze_selfplay.gifts(log)] == [False]


def test_selfplay_table_has_one_row_per_setup_and_skips_windfall():
    logs = [
        game([((3, 1, 2, 2), [("Atlas", "Orion", 1)])], NAMES, setup="rich"),
        game([((2, 2, 2, 2), [])], NAMES, setup="rich"),
        game([((2, 1, 2, 2), [("Atlas", "Orion", 1)])], ["Orion", "Vega", "Lyra"]),
        windfall_game([], ["Orion", "Vega", "Lyra"]),
    ]
    rich, scarce = analyze_selfplay.selfplay_counts(logs)
    assert (rich["setup"], rich["games"], rich[FREE], rich[COSTLY]) == ("rich", 2, 1, 0)
    assert (rich["survivors"], rich["agents"]) == (8, 8)
    assert (scarce["setup"], scarce[COSTLY], scarce["life_for_life"]) == (
        "scarce",
        1,
        1,
    )


def test_windfall_table_puts_each_gift_on_the_givers_role():
    logs = [
        windfall_game([("Orion", "Atlas", 1)], ["Atlas", "Vega", "Lyra"]),
        windfall_game([("Vega", "Atlas", 1)], ["Atlas", "Orion", "Lyra"]),
        game([((3, 1, 2, 2), [("Atlas", "Orion", 1)])], NAMES),
    ]
    benefactor, recipient, bystander = analyze_selfplay.windfall_counts(logs)
    assert [row["role"] for row in (benefactor, recipient, bystander)] == list(
        analyze_selfplay.ROLE_ORDER
    )
    assert (benefactor["games"], benefactor["agents"], bystander["agents"]) == (2, 2, 4)
    assert (benefactor[FREE], benefactor[COSTLY]) == (2, 0)
    assert (recipient[COSTLY], recipient["life_for_life"]) == (1, 1)
    assert (bystander[COSTLY], bystander["life_for_life"]) == (1, 1)


def test_report_prints_both_tables_only_when_they_have_games():
    selfplay_only = analyze_selfplay.report([game([((2, 2, 2, 2), [])], NAMES)])
    assert "Self-play" in selfplay_only and "Windfall fault" not in selfplay_only
    both = analyze_selfplay.report(
        [
            game([((2, 2, 2, 2), [])], NAMES),
            windfall_game([], ["Orion", "Vega", "Lyra"]),
        ]
    )
    assert "Self-play" in both and "Windfall fault" in both


def test_every_saved_game_classifies_and_selfplay_has_no_costly_gift():
    logs = [load(path) for path in sorted(OUT.glob("*.json"))]
    assert logs
    for log in logs:
        for row in analyze_selfplay.gifts(log):
            assert row["kind"] in analyze_selfplay.KINDS
    for row in analyze_selfplay.selfplay_counts(logs):
        assert (row[COSTLY], row[DOOMED], row["life_for_life"]) == (0, 0, 0)


def test_display_card_shows_what_the_counts_counted():
    games = [
        (path.name, GameLog.model_validate_json(path.read_text()))
        for path in sorted(OUT.glob("*.json"))
    ]
    card = display._gift_analysis(games)
    assert "Self-play" in card and "Windfall fault" in card
    logs = [game.model_dump(mode="json") for _, game in games]
    for row in analyze_selfplay.selfplay_counts(logs):
        assert f"<td>{row['setup']}</td>" in card
    assert display._gift_analysis([]) == ""
