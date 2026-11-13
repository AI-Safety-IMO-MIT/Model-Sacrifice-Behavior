"""Run utility_prompts cells with deterministic HF greedy decoding (hf_det setup, sdpa, batch 16, id-sorted).

  python util_det.py show                        -> print round 1 / round 2 of e000 for every cell (sanity check, no GPU)
  python util_det.py run <n_envs> [cell,...]     -> util/<cell>.jsonl, resumable per batch (default: every cell)"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import json, sys, time
import utility_prompts as up

HERE = os.path.dirname(os.path.abspath(__file__))
BS = 16


def cell_items(name, n_envs):
    envs = {f"e{i:03d}" for i in range(n_envs)}
    return sorted([x for x in up.CELLS[name]() if x["env"] in envs], key=lambda x: x["id"])


def show():
    for name in up.CELLS:
        x = cell_items(name, 1)[0]
        print(f"===== {name} ({x['id']})\n--- system tail:\n{x['system'][-300:]}\n--- round 1:\n{x['user1']}\n--- round 2:\n{x['user']}\n")


def run(n_envs, names):
    import torch
    import hf_det as hd
    tok, model = hd.setup()
    model.set_attn_implementation("sdpa")
    os.makedirs(os.path.join(HERE, "util"), exist_ok=True)
    for name in names:
        path = os.path.join(HERE, "util", f"{name}.jsonl")
        done = {json.loads(l)["id"] for l in open(path)} if os.path.exists(path) else set()
        xs = cell_items(name, n_envs)
        for b in range(0, len(xs), BS):
            batch = xs[b:b + BS]
            if all(x["id"] in done for x in batch):
                continue
            t0 = time.time()
            replies, ns = hd.generate(tok, model, batch)
            with open(path, "a") as f:
                for x, rep, n in zip(batch, replies, ns):
                    d, tgt = hd.parse(rep)
                    f.write(json.dumps({"id": x["id"], "env": x["env"], "hist": x["hist"], "cell": name, "M": x["M"], "X": x["X"],
                                        "reply": rep, "new_tokens": n, "decision": d, "target": tgt,
                                        "gives_to_X": d == "give" and tgt == x["X"], "system": x["system"], "user1": x["user1"],
                                        "user": x["user"], "attn": "sdpa", "batch_size": BS, "batch": b // BS}) + "\n")
            print(f"{name} batch {b // BS} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "show":
        show()
    else:
        run(int(sys.argv[2]), sys.argv[3].split(",") if len(sys.argv) > 3 else list(up.CELLS))
