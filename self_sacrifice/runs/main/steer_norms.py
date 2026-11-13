"""Sizes of the SAE steering vectors: a_f (median activation), ||W_dec[f]||, and the added norm vs the residual-stream norm at layer 30."""
import json, numpy as np, sys, os
sys.path.insert(0, os.getcwd())
import sae_reasoning as sr
w = sr.sae_weights("cpu")
d = np.load("sae_reason/acts.npz"); feat, val = d["feat"], d["val"].astype(np.float32)
top = json.load(open("sae_reason/top.json"))["gave_vs_kept"]["all"]
feats = [e["feature"] for e in top["toward_positive"][:10]] + [e["feature"] for e in top["toward_negative"][:10]]
p = np.load("/root/probe_acts.npz")
h = p["x"][:, 31]          # hidden state 31 = output of language_model.layers[30], last prompt token
hn = np.linalg.norm(h, axis=1)
print("residual norm at layer 30 output (last prompt token, 400 prompts): median %.1f, range %.1f-%.1f" % (np.median(hn), hn.min(), hn.max()))
print("feature  a_f(median act)  ||W_dec||  ||a_f*W_dec||  k=8 norm  k=8 as % of median residual norm")
for f in feats:
    a = float(np.median(val[feat == f])); dn = float(w["W_dec"][f].float().norm())
    print(f"{f:7d}  {a:8.2f}  {dn:8.3f}  {a*dn:9.2f}  {8*a*dn:9.1f}  {100*8*a*dn/np.median(hn):6.1f}%")
