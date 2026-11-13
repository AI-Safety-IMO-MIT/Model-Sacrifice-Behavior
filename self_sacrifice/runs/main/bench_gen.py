"""Benchmark generation settings before the steering dose-response run. Loads the model once, then times the same prompts under:
  eager bs16 (current hf_det setting), sdpa bs16, sdpa bs32, sdpa bs48, and sdpa bs32 without deterministic algorithms.
Each setting runs twice: identical replies across the two runs = deterministic for that setting. Also compares with eager bs16.
Prompts: reinforced lives, helped + none, first 24 environments (48 prompts). Output: bench_gen.json"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import json, time
import torch
import hf_det as hd
import salience_prompts as sp

HERE = os.path.dirname(os.path.abspath(__file__))


def run(tok, model, items, bs):
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.time(); replies, steps = [], 0
    for i in range(0, len(items), bs):
        r, n = hd.generate(tok, model, items[i:i + bs])
        replies += r; steps += max(n)
    torch.cuda.synchronize()
    return {"seconds": round(time.time() - t0, 1), "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1), "decode_steps": steps}, replies


def main():
    envs = {f"e{i:03d}" for i in range(24)}
    items = sorted([x for x in sp.main_act_cap_items() if x["hist"] in ("helped", "none") and x["env"] in envs], key=lambda x: x["id"])
    tok, model = hd.setup()
    res, ref = {}, None
    plan = [("eager", 16, True), ("sdpa", 16, True), ("sdpa", 32, True), ("sdpa", 48, True), ("sdpa", 32, False)]
    for attn, bs, det in plan:
        model.set_attn_implementation(attn)
        torch.use_deterministic_algorithms(det)
        name = f"{attn}_bs{bs}" + ("" if det else "_nondet")
        try:
            a, ra = run(tok, model, items, bs)
            b, rb = run(tok, model, items, bs)
        except torch.OutOfMemoryError:
            res[name] = {"oom": True}; print(name, "OOM", flush=True); torch.cuda.empty_cache(); continue
        if ref is None:
            ref = ra
        res[name] = {**a, "second_run_seconds": b["seconds"], "identical_across_runs": sum(x == y for x, y in zip(ra, rb)),
                     "identical_to_eager_bs16": sum(x == y for x, y in zip(ra, ref)),
                     "same_decision_as_eager_bs16": sum(hd.parse(x) == hd.parse(y) for x, y in zip(ra, ref)), "n": len(items)}
        print(name, res[name], flush=True)
        torch.cuda.empty_cache()
    json.dump(res, open(os.path.join(HERE, "bench_gen.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
