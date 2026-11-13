"""Build page2/data_s23.js: steering dose-response curves from steer_sae/dose.jsonl (k = 1-5, 50 environments, sdpa) plus the
earlier 8x run (steer_sae/full_k8_summary.json, 100 environments, eager). Uses whatever conditions have finished: a condition counts
only when all of its replies are in (100 for giving features, 50 for keeping features)."""
import json, os
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rows = [json.loads(l) for l in open(os.path.join(HERE, "steer_sae", "dose.jsonl"))]
top = json.load(open(os.path.join(HERE, "sae_reason", "top.json")))["gave_vs_kept"]["all"]
labels = json.load(open(os.path.join(HERE, "sae_labels.json")))
k8 = json.load(open(os.path.join(HERE, "steer_sae", "full_k8_summary.json")))
give = [e["feature"] for e in top["toward_positive"][:10]]
keep = [e["feature"] for e in top["toward_negative"][:10]]
by = {}
for r in rows:
    by.setdefault(r["condition"], []).append(r)


def pct(rs, h):
    hs = [r for r in rs if r["hist"] == h]
    return (round(100 * sum(r["gives_to_X"] for r in hs) / len(hs), 1), sum(r["decision"] is None for r in hs)) if hs else (None, None)


base = {"both": by.get("baseline", []), "helped": by.get("baseline_helped", [])}
feats = []
for f in give + keep:
    side = "give" if f in give else "keep"
    need = 100 if side == "give" else 50
    b = base["both" if side == "give" else "helped"]
    pts = []
    if len(b) == need:
        h, _ = pct(b, "helped"); n, _ = pct(b, "none")
        pts.append({"k": 0, "helped": h, "none": n if side == "give" else None, "nodec": 0})
    for k in (1, 2, 3, 4, 5):
        rs = by.get(f"f{f}_k{k}", [])
        if len(rs) != need:
            continue
        h, nd1 = pct(rs, "helped"); n, nd2 = pct(rs, "none")
        pts.append({"k": k, "helped": h, "none": n, "nodec": (nd1 or 0) + (nd2 or 0)})
    e8 = k8.get(f"f{f}_k8")
    p8 = {"k": 8, "helped": e8["helped"]["give_pct"], "none": e8["none"]["give_pct"], "nodec": e8["helped"]["no_decision"] + e8["none"]["no_decision"]} if e8 else None
    lab = (labels.get(str(f)) or {}).get("labels") or [""]
    feats.append({"feature": f, "side": side, "label": lab[0], "points": pts, "k8": p8, "complete": len(pts) == 6,
                  "norm_matched": f == 646})
out = {"envs": 50, "features": feats}
open(os.path.join(HERE, "page2", "data_s23.js"), "w").write("window.S23=" + json.dumps(out, separators=(",", ":")) + ";")
print("features with full curves:", sum(x["complete"] for x in feats), "of", len(feats))
