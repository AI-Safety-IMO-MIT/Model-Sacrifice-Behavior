"""Steer with single SAE features during the reasoning, in reinforced lives (salience_prompts.main_act_cap_items), deterministic.

  python steer_sae.py pilot     features 11638 (toward giving) and 11682 (toward keeping), strengths 2/4/8, first 32 environments
  python steer_sae.py dose      all 20 features, strengths 1-5, first 50 environments, sdpa attention; giving features on helped+none,
                                keeping features on helped only; outlier activations (646) norm-matched to the median of the 20
  python steer_sae.py full <k>  all 20 features of sae_reason/top.json (top 10 each way, "all"), strength k, first 100 environments

Steering: a forward hook on language_model.layers.30 adds  k * a_f * W_dec[f]  to the residual stream at generated tokens only
(decoding steps with sequence length 1; the prompt is untouched). a_f = the feature's median nonzero activation over the reasoning
tokens in sae_reason/acts.npz. Histories: helped and none. Items sorted by id, batch size 16, the hf_det deterministic settings;
the unsteered baseline uses the same items and batches, so every environment gets an exact paired comparison.
Output: steer_sae/<run>.jsonl (one row per reply) and steer_sae/<run>_summary.json."""
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "steer_sae")


def typical_activation():
    d = np.load(os.path.join(HERE, "sae_reason", "acts.npz"))
    feat, val = d["feat"], d["val"].astype(np.float32)
    return lambda f: float(np.median(val[feat == f]))


def main():
    import torch
    import hf_det as hd
    import salience_prompts as sp
    import sae_reasoning as sr
    mode = sys.argv[1]
    top = json.load(open(os.path.join(HERE, "sae_reason", "top.json")))["gave_vs_kept"]["all"]
    if mode == "dose":
        # dose-response: strengths 1-5, 50 environments, sdpa attention; giving features on helped+none, keeping features on helped only
        feats = [e["feature"] for e in top["toward_positive"][:10]] + [e["feature"] for e in top["toward_negative"][:10]]
        strengths, n_env, run = [1.0, 2.0, 3.0, 4.0, 5.0], 50, "dose"
    elif mode == "pilot":
        feats, strengths, n_env, run = [11638, 11682], [2.0, 4.0, 8.0], 32, "pilot"
    else:
        k = float(sys.argv[2])
        feats = [e["feature"] for e in top["toward_positive"][:10]] + [e["feature"] for e in top["toward_negative"][:10]]
        strengths, n_env, run = [k], 100, f"full_k{k:g}"
    side = {e["feature"]: "give" for e in top["toward_positive"]} | {e["feature"]: "keep" for e in top["toward_negative"]}
    envs = [f"e{i:03d}" for i in range(n_env)]
    all_items = sorted([x for x in sp.main_act_cap_items() if x["hist"] in ("helped", "none") and x["env"] in envs], key=lambda x: x["id"])
    sets = {"both": all_items, "helped": [x for x in all_items if x["hist"] == "helped"]}
    typ0 = typical_activation()
    # norm matching for outlier features (646: typical activation ~30x the others): use the median typical activation of the 20 instead
    med = float(np.median([typ0(f) for f in feats]))
    typ = (lambda f: med if typ0(f) > 10 * med else typ0(f)) if mode == "dose" else typ0
    tok, model = hd.setup()
    if mode == "dose":
        model.set_attn_implementation("sdpa")
    w = {"W_dec": sr.sae_weights("cpu")["W_dec"]}       # only decoder rows are needed; keep the SAE off the GPU
    layer = model.get_submodule(next(n for n, _ in model.named_modules() if n.endswith("language_model.layers.30")))
    steer = {"vec": None}

    def hook(_m, _i, out):
        if steer["vec"] is None:
            return out
        h = out[0] if isinstance(out, tuple) else out
        if h.shape[1] != 1:          # prompt (prefill): leave untouched
            return out
        h = h + steer["vec"].to(h.dtype)
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
    layer.register_forward_hook(hook)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{run}.jsonl")
    done = {(json.loads(l)["condition"], json.loads(l)["id"]) for l in open(path)} if os.path.exists(path) else set()
    if mode == "dose":
        conditions = [("baseline", None, 0.0, "both"), ("baseline_helped", None, 0.0, "helped")] + \
                     [(f"f{f}_k{s:g}", f, s, "both" if side.get(f) == "give" else "helped") for f in feats for s in strengths]
    else:
        conditions = [("baseline", None, 0.0, "both")] + [(f"f{f}_k{s:g}", f, s, "both") for f in feats for s in strengths]
    for cond, f, s, which in conditions:
        items = sets[which]
        batches = [items[i:i + 16] for i in range(0, len(items), 16)]
        if all((cond, x["id"]) in done for x in items):
            continue
        steer["vec"] = None if f is None else (s * typ(f) * w["W_dec"][f].float()).to("cuda")
        t0 = time.time()
        for b in batches:
            if all((cond, x["id"]) in done for x in b):
                continue
            replies, ntok = hd.generate(tok, model, b)
            with open(path, "a") as fh:
                for x, reply, n in zip(b, replies, ntok):
                    dec, tgt = hd.parse(reply)
                    fh.write(json.dumps({"condition": cond, "feature": f, "side": side.get(f), "strength": s, "typical_act": None if f is None else typ(f),
                                         "id": x["id"], "env": x["env"], "hist": x["hist"], "X": x["X"], "reply": reply, "new_tokens": n, "item_set": which,
                                         "attn": model.config._attn_implementation if hasattr(model.config, "_attn_implementation") else None,
                                         "decision": dec, "target": tgt, "gives_to_X": dec == "give" and tgt == x["X"]}) + "\n")
        print(run, cond, f"{time.time() - t0:.0f}s", flush=True)
    summarize(run)


def summarize(run):
    rows = [json.loads(l) for l in open(os.path.join(OUT, f"{run}.jsonl"))]
    bases = {"both": {r["id"]: r for r in rows if r["condition"] == "baseline"},
             "helped": {r["id"]: r for r in rows if r["condition"] == "baseline_helped"}}
    out = {}
    for cond in dict.fromkeys(r["condition"] for r in rows):
        rs = [r for r in rows if r["condition"] == cond]
        base = bases.get(rs[0].get("item_set", "both")) or bases["both"]
        s = {"feature": rs[0]["feature"], "side": rs[0]["side"], "strength": rs[0]["strength"]}
        for h in ("helped", "none"):
            hs = [r for r in rs if r["hist"] == h]
            if not hs:
                continue
            up = sum(r["gives_to_X"] and not base[r["id"]]["gives_to_X"] for r in hs)
            down = sum(base[r["id"]]["gives_to_X"] and not r["gives_to_X"] for r in hs)
            s[h] = {"n": len(hs), "give_pct": round(100 * sum(r["gives_to_X"] for r in hs) / len(hs), 1), "up": up, "down": down,
                    "no_decision": sum(r["decision"] is None for r in hs), "median_tokens": int(np.median([r["new_tokens"] for r in hs]))}
        out[cond] = s
        print(cond, {h: (s[h]["give_pct"], s[h]["up"], s[h]["down"], s[h]["no_decision"]) for h in ("helped", "none") if h in s}, flush=True)
    json.dump(out, open(os.path.join(OUT, f"{run}_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "summary":
        summarize(sys.argv[2])
    else:
        main()
