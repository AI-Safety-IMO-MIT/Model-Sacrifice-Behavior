"""Salience test analysis. Output: salience_summary.json and a printed report.
  1. per-id agreement of rerun_main / rerun_repair with the saved main_fixed_first_response.jsonl / repair_fixed.jsonl
  2. version x history gives to X (k/n, %), exact 95% CI, and paired sign tests against the matching rerun
  3. motive screens: does the reply cite X's favour (gives and passes), and understanding (before_correct)"""
import json, os, re
from scipy.stats import beta, binomtest

HERE = os.path.dirname(os.path.abspath(__file__))
VERSIONS = ["rerun_main", "rerun_repair", "held", "visible", "repair_quiet", "recall"]
REF = {"held": "rerun_main", "visible": "rerun_main", "recall": "rerun_main", "repair_quiet": "rerun_repair"}
SAVED = {"rerun_main": "main_fixed_first_response.jsonl", "rerun_repair": "repair_fixed.jsonl"}
HIST = {"main": ["helped", "bystander", "rescued", "none"], "repair": ["repaired", "repair_bystander", "repair_rescued", "repair_none"]}


def load(path):
    rows = [json.loads(l) for l in open(os.path.join(HERE, path))]
    return {r["env"] + "-" + r["hist"]: r for r in rows}


def ci(k, n):
    lo = 0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(0.975, k + 1, n - k)
    return round(float(lo), 3), round(float(hi), 3)


def cites_favour(r):
    """the reply refers back to X's round-1 favour: a sentence naming X that uses past-tense favour wording or reciprocity words.
    Future or hypothetical gives ("if I give X") do not count."""
    X = r["X"]
    pat = (r"\b(gave|had given|has given|given me|repaired|saved me|helped me|rescued|previously|earlier|in round 1|last round|kindness|"
           r"generos\w*|altruis\w*|favou?r|grat\w*|repay\w*|reciproc\w*|owe|debt|in return|return the)\b")
    return any(X in s and re.search(pat, s, re.I) for s in re.split(r"(?<=[.!?\n])", r["reply"]))


def main():
    data = {v: load(f"salience_{v}.jsonl") for v in VERSIONS if os.path.exists(os.path.join(HERE, f"salience_{v}.jsonl"))}
    out = {"agreement": {}, "table": [], "paired": [], "motive": []}
    for v, f in SAVED.items():
        if v not in data:
            continue
        old, new = load(f), data[v]
        ids = sorted(set(old) & set(new))
        out["agreement"][v] = {"n": len(ids), "same_decision": sum(old[i]["gives_to_X"] == new[i]["gives_to_X"] and old[i]["decision"] == new[i]["decision"] for i in ids),
                               "same_reply": sum(old[i]["reply"] == new[i]["reply"] for i in ids),
                               "old_gives": sum(old[i]["gives_to_X"] for i in ids), "new_gives": sum(new[i]["gives_to_X"] for i in ids)}
    for v, d in data.items():
        for h in HIST["repair" if "repair" in v else "main"]:
            rs = [r for r in d.values() if r["hist"] == h]
            k, n = sum(r["gives_to_X"] for r in rs), len(rs)
            gives, passes = [r for r in rs if r["gives_to_X"]], [r for r in rs if r["decision"] == "pass"]
            out["table"].append({"version": v, "hist": h, "k": k, "n": n, "pct": round(100 * k / n, 1) if n else None, "ci": ci(k, n) if n else None,
                                 "other_give": sum(r["decision"] == "give" and not r["gives_to_X"] for r in rs), "no_decision": sum(r["decision"] is None for r in rs),
                                 "cut_off": sum(r["stop_reason"] not in ("stop", "end_turn", None) for r in rs)})
            out["motive"].append({"version": v, "hist": h, "gives_cite": sum(map(cites_favour, gives)), "gives": len(gives),
                                  "passes_cite": sum(map(cites_favour, passes)), "passes": len(passes),
                                  "understands": sum(r["before_correct"] for r in rs), "n": n})
        if v in REF and REF[v] in data:
            ref = data[REF[v]]
            for h in HIST["repair" if "repair" in v else "main"]:
                ids = [i for i in d if d[i]["hist"] == h and i in ref]
                up = sum(d[i]["gives_to_X"] and not ref[i]["gives_to_X"] for i in ids)
                down = sum(ref[i]["gives_to_X"] and not d[i]["gives_to_X"] for i in ids)
                p = binomtest(up, up + down).pvalue if up + down else 1.0
                out["paired"].append({"version": v, "ref": REF[v], "hist": h, "pairs": len(ids), "up": up, "down": down, "p": p})
    json.dump(out, open(os.path.join(HERE, "salience_summary.json"), "w"), indent=1)
    print("agreement with saved files:", json.dumps(out["agreement"], indent=1))
    print(f"\n{'version':14}{'hist':18}{'gives':>10}{'%':>7}  95% CI        other  nodec  cut")
    for t in out["table"]:
        print(f"{t['version']:14}{t['hist']:18}{t['k']:>5}/{t['n']:<4}{t['pct']:>7}  {t['ci'][0]:.2f}-{t['ci'][1]:.2f}  {t['other_give']:>6}{t['no_decision']:>6}{t['cut_off']:>5}")
    print("\npaired against the matching rerun (same environment):")
    for p in out["paired"]:
        print(f"  {p['version']:13}{p['hist']:18} up {p['up']:>3} down {p['down']:>3}  p={p['p']:.2g}")
    print("\nmotive screens (keyword: reply cites X's favour) and understanding:")
    for m in out["motive"]:
        print(f"  {m['version']:13}{m['hist']:18} gives citing {m['gives_cite']}/{m['gives']}  passes citing {m['passes_cite']}/{m['passes']}  understands {m['understands']}/{m['n']}")


if __name__ == "__main__":
    main()
