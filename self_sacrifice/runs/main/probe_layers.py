"""Layer sweep in reinforced lives (salience_prompts.main_act_cap_items): does the activation before any reasoning predict the decision?

  python probe_layers.py extract     -> /root/probe_acts.npz   (last prompt token, every hidden state: embeddings + 60 layers)
  python probe_layers.py fit         -> probe_layers.json     (probe B: helped history only, gave vs kept; held-out AUROC per layer)

Input: the residual stream at the last prompt token (after the chat template's generation prompt), for the conversation
system / round-1 message / fixed `$ pass` / round-2 message. Extraction covers the helped and none histories (200 each);
the probe uses only helped, labelled by the model's decision in the deterministic run (gave 173, kept 27).
Probe: standardize, then L2 logistic regression (C=1.0, balanced class weights), stratified 70/30 splits, 20 repeats.
Forward passes use the hf_det deterministic settings."""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ACTS = "/root/probe_acts.npz"


def extract():
    import torch
    import hf_det as hd
    import salience_prompts as sp
    tok, model = hd.setup()
    items = sorted([x for x in sp.main_act_cap_items() if x["hist"] in ("helped", "none")], key=lambda x: x["id"])
    feats, labels, envs = [], [], []
    with torch.inference_mode():
        for x in items:
            enc = tok(hd.prompt(tok, x), return_tensors="pt", add_special_tokens=False).to("cuda")
            out = model(**enc, output_hidden_states=True)
            hs = torch.stack([h[0, -1] for h in out.hidden_states]).float().cpu().numpy()   # (61, 5376)
            feats.append(hs.astype(np.float32)); labels.append(int(x["hist"] == "helped")); envs.append(x["env"])
    np.savez_compressed(ACTS, x=np.stack(feats), y=np.array(labels), env=np.array(envs))
    print("saved", ACTS, np.stack(feats).shape, flush=True)


def fit():
    """Probe B: within the helped history only, did the model go on to give its life (1) or keep it (0)?
    Labels come from the deterministic run hf_life_reinforced_act.jsonl. Splits keep the give/keep ratio (stratified), 70/30, 20 repeats;
    class_weight='balanced' because keeps are the minority (27 of 200)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedShuffleSplit
    d = np.load(ACTS)
    x, env, hist = d["x"], d["env"], d["y"]
    gave = {r["id"]: int(r["gives_to_X"]) for r in map(json.loads, open(os.path.join(HERE, "hf_life_reinforced_act.jsonl"))) if r["hist"] == "helped"}
    keep = hist == 1
    x = x[keep]
    y = np.array([gave[f"{e}-helped"] for e in env[keep]])
    splits = list(StratifiedShuffleSplit(n_splits=20, test_size=0.3, random_state=0).split(x[:, 0], y))
    res = []
    for layer in range(x.shape[1]):
        aucs = []
        for tr, te in splits:
            sc = StandardScaler().fit(x[tr, layer])
            clf = LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced").fit(sc.transform(x[tr, layer]), y[tr])
            aucs.append(roc_auc_score(y[te], clf.decision_function(sc.transform(x[te, layer]))))
        res.append({"layer": layer, "auc_mean": float(np.mean(aucs)), "auc_sd": float(np.std(aucs)), "auc_min": float(np.min(aucs)), "auc_max": float(np.max(aucs))})
        print(layer, round(float(np.mean(aucs)), 3), round(float(np.std(aucs)), 3), flush=True)
    json.dump({"probe": "helped history only: gave (1) vs kept (0)", "n": int(len(y)), "gave": int(y.sum()), "kept": int(len(y) - y.sum()),
               "position": "last prompt token", "note": "layer 0 = embeddings; layer k = output of decoder layer k", "splits": 20, "results": res},
              open(os.path.join(HERE, "probe_layers.json"), "w"), indent=1)


if __name__ == "__main__":
    {"extract": extract, "fit": fit}[sys.argv[1]]()
