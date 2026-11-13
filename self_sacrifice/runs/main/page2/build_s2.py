"""Build page2/data_s2.js for section 2 (SAE features during reasoning, gave vs kept, reinforced lives helped history)."""
import json, numpy as np, re, os
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(HERE)
t = json.load(open('sae_reason/top.json')); L = json.load(open('sae_labels.json'))
m = json.load(open('sae_reason/meta.json')); d = np.load('sae_reason/acts.npz')
reps, toks = m['replies'], m['tokens']
hf = {r['id']: r for r in map(json.loads, open('hf_life_reinforced_act.jsonl'))}
G = t['gave_vs_kept']['all']
feats = [(e, 'give') for e in G['toward_positive'][:10]] + [(e, 'keep') for e in G['toward_negative'][:10]]
def clean(tk):
    return tk.replace('▁', ' ').replace('<0x0A>', '\n')
replies = {}
for i, r in enumerate(reps):
    if r['hist'] != 'helped':
        continue
    h = hf[r['id']]
    cmds = list(re.finditer(r"\$\s*(?:give\s+\w+\s+\d+|pass\b)", h['reply'], re.I))
    replies[r['id']] = {"env": r['env'], "X": h['X'], "M": h['M'], "gave": r['gave'], "toks": [clean(x) for x in toks[i]],
                        "end": h['reply'][cmds[-1].start():].strip() if cmds else ""}
rep_arr, tok_arr, feat_arr, val_arr = d['rep'], d['tok'], d['feat'], d['val'].astype(float)
out = []
for e, side in feats:
    f = e['feature']; sel = feat_arr == f
    per = {}
    for ri, ti, v in zip(rep_arr[sel], tok_arr[sel], val_arr[sel]):
        if reps[ri]['hist'] != 'helped':
            continue
        per.setdefault(reps[ri]['id'], []).append([int(ti), round(float(v), 2)])
    order = lambda ids: sorted(ids, key=lambda k: -sum(v for _, v in per[k]) / len(replies[k]['toks']))
    lab = (L.get(str(f)) or {}).get('labels') or ['']
    out.append({"feature": f, "side": side, "auroc": e['auroc'], "label": lab[0], "labels": lab, "context": e['contexts'][0],
                "vmax": max(v for k in per for _, v in per[k]), "gave": order([k for k in per if replies[k]['gave']]),
                "kept": order([k for k in per if not replies[k]['gave']]), "acts": per})
used = set(k for o in out for k in o['acts'])
data = {"replies": {k: v for k, v in replies.items() if k in used}, "features": out}
open('page2/data_s2.js', 'w').write('window.S2=' + json.dumps(data, separators=(',', ':')) + ';')
print('size', os.path.getsize('page2/data_s2.js'), 'replies', len(data['replies']))
for o in out:
    print(o['feature'], o['side'], o['auroc'], len(o['gave']), len(o['kept']), o['label'][:40])
first = next(iter(data['replies'].values()))
print(repr(''.join(first['toks'][:50])), '|', first['end'])
