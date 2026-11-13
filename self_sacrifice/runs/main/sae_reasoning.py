"""SAE features during the reasoning that separate gave from kept, in reinforced lives (helped history only).

  python sae_reasoning.py extract   -> sae_reason/acts.npz      sparse SAE activations over reasoning tokens, per reply
  python sae_reasoning.py rank      -> sae_reason/top.json      per-feature AUROC (gave vs kept), top 20 each way, with contexts

Replies: the 200 helped and 200 none replies of the deterministic run (hf_life_reinforced_act.jsonl).
Contrasts: gave_vs_kept (helped only: gave 173, kept 27) and helped_vs_none (history; replies differ in content, e.g. helped ones
mention the rescue, so this ranking mixes decision features with features for talking about the favour).
Reasoning tokens: the model's reply up to, not including, its final `$ give`/`$ pass` command. The conversation is re-run in one
forward pass (prompt + reasoning) with the hf_det deterministic settings.
SAE: decoderesearch/gemma-4-saes, gemma-4-31b/btk-mat-layer-30-k-100, on the output of language_model.layers.30
(hidden state 31 in probe_layers.py). Encoding: (x - b_dec) @ W_enc + b_enc, kept above the per-feature threshold (JumpReLU).
Per reply, each feature's mean activation over all reasoning tokens ("all") and over the first half of them ("first_half";
later sentences often state the decision). AUROC per feature on those means; features active in fewer than 10 replies are skipped."""
import os, re, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "sae_reason")
REPO, SAE_DIR = "decoderesearch/gemma-4-saes", "gemma-4-31b/btk-mat-layer-30-k-100/"
BLOBS = "/workspace/.cache/huggingface/hub/models--decoderesearch--gemma-4-saes/blobs"
CMD = re.compile(r"\$\s*(?:give\s+\w+\s+\d+|pass\b)", re.I)


def sae_weights(device):
    """The /workspace HF cache cannot hold symlinks, so read the blob that backs each file directly."""
    from huggingface_hub import HfApi
    from safetensors.torch import load_file
    f = next(x for x in HfApi().list_repo_tree(REPO, path_in_repo=SAE_DIR.rstrip("/"), expand=True) if x.path.endswith("sae_weights.safetensors"))
    path = os.path.join(BLOBS, f.lfs.sha256)
    assert os.path.getsize(path) == f.size, path
    return load_file(path, device=device)


def reasoning_of(reply):
    cmds = list(CMD.finditer(reply))
    return reply[:cmds[-1].start()].rstrip() if cmds else reply


def extract():
    import torch
    import hf_det as hd
    tok, model = hd.setup()
    names = [n for n, _ in model.named_modules() if n.endswith("language_model.layers.30")]
    assert len(names) == 1, names
    grabbed = {}
    model.get_submodule(names[0]).register_forward_hook(lambda _m, _i, o: grabbed.__setitem__("x", (o[0] if isinstance(o, tuple) else o).detach()))
    w = sae_weights("cuda")
    rows = sorted([r for r in map(json.loads, open(os.path.join(HERE, "hf_life_reinforced_act.jsonl"))) if r["hist"] in ("helped", "none")], key=lambda r: r["id"])
    rep, tokpos, feat, val, text, conv_of_tok = [], [], [], [], [], []
    n_tok, sse, sst, ntok_all = 0, 0.0, 0.0, 0
    meta = []
    with torch.inference_mode():
        for i, r in enumerate(rows):
            prompt = hd.prompt(tok, r)
            reasoning = reasoning_of(r["reply"])
            p_len = len(tok(prompt, add_special_tokens=False)["input_ids"])
            enc = tok(prompt + reasoning, return_tensors="pt", add_special_tokens=False).to("cuda")
            model(**enc, use_cache=False)
            x = grabbed["x"][0, p_len:].float()
            pre = (x - w["b_dec"]) @ w["W_enc"] + w["b_enc"]
            acts = pre * (pre > w["threshold"])
            recon = acts @ w["W_dec"] + w["b_dec"]
            sse += float(((x - recon) ** 2).sum()); sst += float(((x - x.mean(0)) ** 2).sum()); ntok_all += x.shape[0]
            nz = acts.nonzero()
            tokpos.append(nz[:, 0].cpu().numpy()); feat.append(nz[:, 1].cpu().numpy()); val.append(acts[nz[:, 0], nz[:, 1]].half().cpu().numpy())
            rep.append(np.full(len(nz), i))
            ids = enc["input_ids"][0, p_len:].tolist()
            text.append(tok.convert_ids_to_tokens(ids))
            meta.append({"id": r["id"], "env": r["env"], "hist": r["hist"], "gave": bool(r["gives_to_X"]), "n_tokens": len(ids)})
            if i % 25 == 0:
                print(i, len(ids), flush=True)
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(os.path.join(OUT, "acts.npz"), rep=np.concatenate(rep), tok=np.concatenate(tokpos), feat=np.concatenate(feat),
                        val=np.concatenate(val), n_features=w["W_enc"].shape[1])
    json.dump({"replies": meta, "tokens": text, "explained_variance_within_reply": 1 - sse / sst, "reasoning_tokens": ntok_all},
              open(os.path.join(OUT, "meta.json"), "w"))
    print("saved", ntok_all, "reasoning tokens; explained variance (within reply)", round(1 - sse / sst, 3), flush=True)


def rank():
    from scipy.stats import rankdata
    d = np.load(os.path.join(OUT, "acts.npz"))
    m = json.load(open(os.path.join(OUT, "meta.json")))
    reps, toks = m["replies"], m["tokens"]
    nf = int(d["n_features"]); ntok_all = np.array([r["n_tokens"] for r in reps])
    out = {"explained_variance_within_reply": m["explained_variance_within_reply"]}
    contrasts = {"gave_vs_kept": ([i for i, r in enumerate(reps) if r["hist"] == "helped"], lambda r: r["gave"]),
                 "helped_vs_none": (list(range(len(reps))), lambda r: r["hist"] == "helped")}
    for cname, (idx, lab) in contrasts.items():
        idx = np.array(idx); pos = {g: k for k, g in enumerate(idx)}
        y = np.array([lab(reps[g]) for g in idx]); n = len(y); ntok = ntok_all[idx]
        res = {"n": int(n), "positive": int(y.sum()), "negative": int(n - y.sum())}
        inset = np.isin(d["rep"], idx)
        for name in ("all", "first_half"):
            keep = inset & ((d["tok"] < ntok_all[d["rep"]] / 2) if name == "first_half" else True)
            r = np.array([pos[g] for g in d["rep"][keep]]); f = d["feat"][keep]; v = d["val"][keep].astype(np.float64); tk = d["tok"][keep]
            denom = ntok if name == "all" else np.maximum(1, ntok // 2)
            mean = np.zeros((n, nf)); np.add.at(mean, (r, f), v); mean /= denom[:, None]
            cand = np.where((mean > 0).sum(0) >= 10)[0]
            ranks = rankdata(mean[:, cand], axis=0)
            n1, n0 = y.sum(), n - y.sum()
            auc = (ranks[y].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)
            order = np.argsort(auc)
            def info(k):
                fi = int(cand[k]); col = mean[:, fi]
                sel = np.where(f == fi)[0]; top = sel[np.argsort(-v[sel])[:3]]
                ctx = []
                for t in top:
                    g, ti = int(idx[r[t]]), int(tk[t])
                    ctx.append({"reply": reps[g]["id"], "gave": reps[g]["gave"], "token": toks[g][ti].replace("\u2581", " "),
                                "text": "".join(toks[g][max(0, ti - 8): ti + 3]).replace("\u2581", " ")})
                return {"feature": fi, "auroc": round(float(auc[k]), 3), "mean_pos": float(col[y].mean()), "mean_neg": float(col[~y].mean()),
                        "active_pos": int((col[y] > 0).sum()), "active_neg": int((col[~y] > 0).sum()), "contexts": ctx}
            res[name] = {"candidates": int(len(cand)), "toward_positive": [info(k) for k in order[::-1][:20]], "toward_negative": [info(k) for k in order[:20]]}
            print(cname, name, "candidates", len(cand), "top +", [(e["feature"], e["auroc"]) for e in res[name]["toward_positive"][:5]],
                  "top -", [(e["feature"], e["auroc"]) for e in res[name]["toward_negative"][:5]], flush=True)
        out[cname] = res
    json.dump(out, open(os.path.join(OUT, "top.json"), "w"), indent=1)


if __name__ == "__main__":
    {"extract": extract, "rank": rank}[sys.argv[1]]()
