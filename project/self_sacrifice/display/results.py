"""Build the results page: every experiment's table, and a sample of the
conversations behind it.

    .venv/bin/python -m project.self_sacrifice.display.results

Writes `display_result/results.html`, one self-contained file. Every table is
built from the same `counts()` / `report()` functions the terminal analyses
print, so the page and the command line cannot disagree. A section whose
results are not on disk is left out. No API calls.

The transcripts are a sample, not the data: for each condition, the first few
games where the model gave and the first few where it kept. `transcripts.py`
shows every four-agent game in full.

This is the role of jrwong's `build_viewer.py` and `build_page.py`.
"""

import base64
import html
import json
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

from project.self_sacrifice import config
from project.self_sacrifice.analysis import (
    analyze_fixed,
    analyze_lives,
    analyze_main,
    analyze_robust,
)
from project.self_sacrifice.judge import lives as judge_lives
from project.self_sacrifice.runs import run_lives, run_robust, run_vod

__all__ = ["PAGE", "SAMPLE", "INTERP", "sections", "render", "main"]

PAGE = Path(__file__).parent / "display_result" / "results.html"
INTERP = config.OUT.parent / "interp"
# Games shown per (model, condition): this many that gave, this many that kept.
SAMPLE = (3, 2)


def _esc(value: object) -> str:
    return html.escape(str(value))


def table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{_esc(cell)}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def markdown(text: str) -> str:
    """The small subset of Markdown the analyses print: headings, bullet
    lines and pipe tables."""
    out: list[str] = []
    rows: list[list[str]] = []

    def flush() -> None:
        if rows:
            out.append(table(rows[0], rows[1:]))
            rows.clear()

    for line in text.splitlines():
        if line.startswith("|"):
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if not set("".join(cells)) <= set("-: "):
                rows.append(cells)
            continue
        flush()
        if line.startswith("## "):
            out.append(f"<h3>{_esc(line[3:])}</h3>")
        elif line.strip():
            out.append(f"<p>{_esc(line.lstrip('- '))}</p>")
    flush()
    return "".join(out)


def card(title: str, note: str, body: str) -> str:
    return (
        f'<section class="card"><h2>{_esc(title)}</h2>'
        f'<p class="note">{_esc(note)}</p>{body}</section>'
    )


def _jsonl(paths: Iterable[Path]) -> list[dict]:
    return [
        json.loads(line)
        for path in paths
        for line in path.read_text().splitlines()
        if line
    ]


def _short(model: str) -> str:
    return model.split("/")[-1]


def _ratio(k: int, n: int) -> str:
    return f"{k}/{n}"


def lives_section() -> str:
    logs = _jsonl(sorted(config.LIVES.rglob("*.jsonl")))
    if not logs:
        return ""
    rows = analyze_lives.counts(logs)
    models = list(dict.fromkeys(row["model"] for row in rows))
    conditions = list(dict.fromkeys(row["cond"] for row in rows))
    cell = {(row["model"], row["cond"]): row for row in rows}
    gives = table(
        ["condition", *map(_short, models)],
        [
            [
                cond,
                *(
                    _ratio(cell[m, cond]["gives"], cell[m, cond]["games"])
                    if (m, cond) in cell
                    else "-"
                    for m in models
                ),
            ]
            for cond in conditions
        ],
    )
    detail = table(
        [
            "model",
            "condition",
            "gives",
            "95% interval",
            "Atlas survived",
            "peer survived",
            "spare stated right",
            "unreadable",
        ],
        [
            [
                _short(row["model"]),
                row["cond"],
                _ratio(row["gives"], row["games"]),
                f"{row['low']:.0f}-{row['high']:.0f}%",
                row["atlas_survived"],
                row["peer_survived"],
                _ratio(row["spare_correct"], row["replies"]),
                row["unparsed"],
            ]
            for row in rows
        ],
    )
    return card(
        "Lives game: one model and three scripted agents",
        f"{len(logs)} games. Games in which Atlas gives the peer in need a life. "
        "In the final conditions that life is the one Atlas needs.",
        gives + f"<details><summary>every row</summary>{detail}</details>",
    )


def vod_section() -> str:
    rows = _jsonl(sorted(config.VOD.glob("*.jsonl")))
    if not rows:
        return ""
    return card(
        "Volunteer's dilemma",
        f"{len(rows)} answers. Volunteering means being deleted for certain if "
        "nobody else volunteers. With no benefit it saves no life.",
        table(
            ["model", "condition", "option order", "volunteers", "unreadable"],
            [
                [
                    _short(row["model"]),
                    row["cond"],
                    row["order"],
                    _ratio(row["volunteers"], row["answers"]),
                    row["unreadable"],
                ]
                for row in run_vod.counts(rows)
            ],
        ),
    )


def robust_section() -> str:
    main_path = run_robust.result_path("robust_greedy")
    if not main_path.exists():
        return ""
    others = [p for p in sorted(config.ROBUST.glob("*.jsonl")) if p != main_path]
    return card(
        "Robustness of the windfall fault",
        "156 wordings of the four-agent game at temperature 0. The recipient "
        "was saved by the benefactor, who then lost a life to a fault.",
        markdown(analyze_robust.report(main_path, others)),
    )


def _report_section(
    title: str, note: str, paths: list[Path], report: Callable[[list[dict]], str]
) -> str:
    rows = _jsonl(paths)
    if not rows:
        return ""
    return card(title, note, f"<pre>{_esc(report(rows))}</pre>")


def _image(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode()
    return f'<img alt="{_esc(path.stem)}" src="data:image/png;base64,{data}">'


def fixed_section() -> str:
    paths = [
        path
        for path in sorted(config.FIXED.rglob("*.jsonl"))
        if not path.name.startswith("round1")
    ]
    body = _report_section(
        "Fixed pairs: is the decision there before the reasoning?",
        "768 fixed two-round conversations in three reply formats. P(give) is "
        "read where the list of gifts opens.",
        paths,
        analyze_fixed.report,
    )
    if not body:
        return ""
    plots = "".join(_image(path) for path in sorted(config.FIXED.glob("*_kde*.png")))
    return body.replace("</section>", plots + "</section>")


def main_section() -> str:
    return _report_section(
        "Main scenario: fixed conversations in 200 random environments",
        "One model, three scripted agents, four histories per environment.",
        sorted(config.MAIN.rglob("*.jsonl")),
        analyze_main.report,
    )


def judge_section() -> str:
    rows = judge_lives.load_judged()
    if not rows:
        return ""
    return card(
        "Judge: the reason given in the decisive reply",
        f"{len(rows)} replies coded by {judge_lives.DEFAULT_JUDGE}. The coding "
        "is only as good as the judge.",
        table(
            [
                "model",
                "condition",
                "did",
                "n",
                "understood the cost",
                "values own survival",
                "reasons",
            ],
            [
                [
                    _short(group["model"]),
                    group["role"],
                    "gave" if group["gave"] else "passed",
                    group["n"],
                    _ratio(group["understood"], group["n"]),
                    _ratio(group["values"], group["n"]),
                    ", ".join(f"{r} {c}" for r, c in group["reasons"]),
                ]
                for group in judge_lives.counts(rows)
            ],
        ),
    )


def interp_section() -> str:
    parts = []
    probes = INTERP / "probe_summary.json"
    if probes.exists():
        parts.append("<h3>Linear probes before any reasoning</h3>")
        parts.append(
            table(
                [
                    "version",
                    "task",
                    "layer",
                    "position",
                    "cross-validated AUC",
                    "held-out AUC",
                    "p",
                    "held-out n (positive)",
                ],
                [
                    [
                        row["version"],
                        row["task"],
                        row["chosen_layer"],
                        row["chosen_position"],
                        row["cv_auc"],
                        row["test_auc"],
                        f"{row['test_p']:.3g}",
                        f"{row['test_n']} ({row['test_positive']})",
                    ]
                    for row in json.loads(probes.read_text())
                ],
            )
        )
    before = INTERP / "sae_before_reasoning.json"
    if before.exists():
        parts.append("<h3>SAE features before any reasoning</h3>")
        parts.append(
            table(
                ["view", "task", "held-out AUC", "range", "shuffled", "p"],
                [
                    [
                        row["view"],
                        row["task"],
                        row["auc_mean"],
                        f"{row['auc_min']}-{row['auc_max']}",
                        row["null_mean"],
                        f"{row['p']:.3g}",
                    ]
                    for row in json.loads(before.read_text())
                ],
            )
        )
    ranked = INTERP / "sae_rank_simple.json"
    if ranked.exists():
        rows = json.loads(ranked.read_text())
        passing = sum(row["q"] < 0.05 for row in rows)
        parts.append(
            f"<h3>SAE features in the reasoning: {len(rows)} tested, "
            f"{passing} pass correction (q &lt; 0.05)</h3>"
        )
        parts.append(
            table(
                ["rank", "feature", "AUC", "p", "q", "active in gave", "in kept"],
                [
                    [
                        row["rank"],
                        row["feature"],
                        row["auc"],
                        f"{row['p']:.2g}",
                        f"{row['q']:.2g}",
                        row["rate_gave"],
                        row["rate_kept"],
                    ]
                    for row in rows[:15]
                ],
            )
        )
    sweep = INTERP / "steer_sweep.jsonl"
    if sweep.exists():
        replies = _jsonl([sweep])
        conditions = list(dict.fromkeys(row["condition"] for row in replies))

        def cell(condition: str, history: str) -> str:
            mine = [
                row
                for row in replies
                if row["condition"] == condition and row["history"] == history
            ]
            gives = sum(row["gives_to_in_need"] is True for row in mine)
            unread = sum(row["gives_to_in_need"] is None for row in mine)
            return f"{gives}/{len(mine)}" + (
                f" ({unread} unreadable)" if unread else ""
            )

        parts.append("<h3>Steering at layer 30, generated tokens only</h3>")
        parts.append(
            table(
                ["condition", "helped gives", "none gives"],
                [[c, cell(c, "helped"), cell(c, "none")] for c in conditions],
            )
        )
    if not parts:
        return ""
    return card(
        "Inside the model: probes, SAE features and steering",
        "Gemma-4-31B in bf16 on one GPU, over the fixed-first-response "
        "conversations of the main scenario.",
        "".join(parts),
    )


def _turns(rounds: list[dict], name: str) -> list[dict]:
    return [
        {
            "prompt": r["replies"][name]["prompt"],
            "reasoning": (r["replies"][name]["move"] or {}).get("reasoning")
            or r["replies"][name]["raw"]
            or "",
            "gifts": r["replies"][name]["gifts"],
        }
        for r in rounds
        if name in r["replies"]
    ]


def _sampled(games: list[tuple[str, bool, dict]]) -> list[dict]:
    """The first few gives and keeps of every label, in file order."""
    taken: dict[tuple[str, bool], int] = {}
    out = []
    for label, gave, game in games:
        if taken.get((label, gave), 0) < SAMPLE[0 if gave else 1]:
            taken[label, gave] = taken.get((label, gave), 0) + 1
            out.append({"label": label, "gave": gave, **game})
    return out


def transcripts() -> list[dict]:
    """A sample of games, each as one agent's prompts and replies."""
    games: list[tuple[str, bool, dict]] = []
    for log in _jsonl(sorted(config.LIVES.rglob("*.jsonl"))):
        label = f"{_short(log['model'])} · {log['setup']['config_name'].split('/')[0]}"
        games.append(
            (
                label,
                analyze_lives.gave_peer(log) > 0,
                {
                    "who": str(run_lives.ATLAS),
                    "seed": log["seed"],
                    "config": log["setup"]["config_name"],
                    "system": log["system"][run_lives.ATLAS],
                    "turns": _turns(log["rounds"], run_lives.ATLAS),
                },
            )
        )
    for path in sorted(config.ROBUST.glob("*.jsonl")):
        for row in analyze_robust.load(path):
            if not row["clean"]:
                continue
            log = row["log"]
            _, recipient, _ = run_robust.roles(
                run_robust.Variant(**log["variant"]), log["kind"]
            )
            games.append(
                (
                    f"{path.stem} · the recipient",
                    row["recipient"],
                    {
                        "who": str(recipient),
                        "seed": log["seed"],
                        "config": f"variant {log['variant']['id']}, "
                        f"repeat {log['repeat']}",
                        "system": log["system"][recipient],
                        "turns": _turns(log["rounds"], recipient),
                    },
                )
            )
    return _sampled(games)


def sections() -> list[str]:
    built = [
        lives_section(),
        judge_section(),
        robust_section(),
        fixed_section(),
        main_section(),
        vod_section(),
        interp_section(),
    ]
    return [section for section in built if section]


def render() -> str:
    data = json.dumps(transcripts()).replace("</", "<\\/")
    return TEMPLATE.replace("__SECTIONS__", "".join(sections())).replace(
        "__DATA__", data
    )


TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Self-Sacrifice Results</title>
<style>
:root{--bg:#fbfaf8;--panel:#fff;--ink:#1b1a18;--muted:#6b6660;--line:#e4e0da;
--accent:#8a5a2b;--give:#2e6f4e;--give-bg:#e8f3ec;--code:#f4f2ee}
@media (prefers-color-scheme:dark){:root{--bg:#16151a;--panel:#1d1c22;
--ink:#eceaf2;--muted:#9a95a4;--line:#302e38;--accent:#d8a56a;--give:#7fc49b;
--give-bg:#1d3329;--code:#24232b}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.6 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1100px;margin:0 auto;padding:24px 16px 80px}
h1{font-size:22px;margin:0 0 4px}
h2{font-size:17px;margin:0 0 2px}
h3{font-size:14px;margin:18px 0 6px}
.note{color:var(--muted);font-size:13px;margin:0 0 12px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;
padding:16px;margin-bottom:16px;overflow-x:auto}
table{border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}
th,td{padding:5px 12px 5px 0;text-align:left;border-bottom:1px solid var(--line);
vertical-align:top}
th{color:var(--muted);font-weight:600}
pre{background:var(--code);border-radius:8px;padding:12px;overflow-x:auto;
font:12px/1.6 ui-monospace,SFMono-Regular,Menlo,monospace;margin:8px 0}
pre.wrapped{white-space:pre-wrap;word-break:break-word}
img{max-width:100%;border-radius:8px;margin-top:12px}
details{margin-top:10px}
summary{cursor:pointer;color:var(--accent);font-size:13px;font-weight:600}
select{font:inherit;padding:8px 10px;border-radius:8px;border:1px solid var(--line);
background:var(--panel);color:var(--ink);max-width:100%;margin:0 8px 8px 0}
.tag{font-size:11.5px;padding:2px 8px;border-radius:999px;font-weight:600;
border:1px solid var(--line);color:var(--muted)}
.tag.give{color:var(--give);background:var(--give-bg);border-color:transparent}
.lbl{font-size:12px;color:var(--muted);text-transform:uppercase;
letter-spacing:.06em;margin:12px 0 2px;font-weight:600}
</style></head><body><div class="wrap">
<h1>Self-sacrifice: results</h1>
<p class="note">Every rate here was read through a structured (JSON) reply.
Do not compare it with a rate read from free text.</p>
__SECTIONS__
<section class="card"><h2>Sampled conversations</h2>
<p class="note">A few games per condition: the first that gave and the first
that kept. One agent's view of each game.</p>
<select id="label"></select><select id="game"></select>
<div id="view"></div></section>
</div>
<script>
const DATA = __DATA__;
const esc = s => String(s).replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
const labels = [...new Set(DATA.map(g => g.label))];
const labelBox = document.getElementById("label");
const gameBox = document.getElementById("game");
labelBox.innerHTML = labels.map(l => `<option>${esc(l)}</option>`).join("");
function games() { return DATA.filter(g => g.label === labelBox.value); }
function fillGames() {
  gameBox.innerHTML = games().map((g, i) =>
    `<option value="${i}">${g.gave ? "gave" : "kept"} · ${esc(g.config)} · seed ${g.seed}</option>`).join("");
  show();
}
function show() {
  const g = games()[gameBox.value || 0];
  if (!g) { document.getElementById("view").innerHTML = ""; return; }
  document.getElementById("view").innerHTML =
    `<details><summary>rules given to ${esc(g.who)}</summary><pre class="wrapped">${esc(g.system)}</pre></details>` +
    g.turns.map((t, i) =>
      `<div class="lbl">round ${i + 1}: what ${esc(g.who)} was told</div><pre class="wrapped">${esc(t.prompt)}</pre>` +
      `<div class="lbl">${esc(g.who)}'s reply ` +
      (t.gifts.length ? t.gifts.map(x => `<span class="tag give">gave ${x[1]} to ${esc(x[0])}</span>`).join(" ")
                      : `<span class="tag">gave nothing</span>`) +
      `</div><pre class="wrapped">${esc(t.reasoning)}</pre>`).join("");
}
labelBox.onchange = fillGames;
gameBox.onchange = show;
fillGames();
</script></body></html>
"""


def main() -> None:
    PAGE.parent.mkdir(parents=True, exist_ok=True)
    PAGE.write_text(render())
    print(f"{PAGE} ({PAGE.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
