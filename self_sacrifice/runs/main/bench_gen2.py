"""Second generation benchmark (sdpa throughout): where the memory goes, and per-step overhead.
  1. prompt pass only (one forward, logits for the last position) for 16 prompts: peak memory
  2. sdpa bs16 with deterministic algorithms off: time
  3. sdpa bs24: time and memory
  4. static cache + torch.compile, bs16: time (first run includes compilation), determinism across two runs, agreement with plain sdpa
Prompts: reinforced lives, helped + none, first 24 environments (48 prompts). Output: bench_gen2.json"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import json, time
import torch
import hf_det as hd
import salience_prompts as sp

HERE = os.path.dirname(os.path.abspath(__file__))


def timed(fn):
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.time()
    out = fn(); torch.cuda.synchronize()
    return out, round(time.time() - t0, 1), round(torch.cuda.max_memory_allocated() / 2**30, 1)


def gen_all(tok, model, items, bs, **kw):
    reps, steps = [], 0
    for i in range(0, len(items), bs):
        b = items[i:i + bs]
        enc = tok([hd.prompt(tok, x) for x in b], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.inference_mode():
            out = model.generate(**enc, do_sample=False, max_new_tokens=hd.MAX_NEW, pad_token_id=tok.pad_token_id, **kw)
        new = out[:, enc["input_ids"].shape[1]:]
        reps += [tok.decode(s, skip_special_tokens=True).strip() for s in new]; steps += new.shape[1]
    return reps, steps


def main():
    envs = {f"e{i:03d}" for i in range(24)}
    items = sorted([x for x in sp.main_act_cap_items() if x["hist"] in ("helped", "none") and x["env"] in envs], key=lambda x: x["id"])
    tok, model = hd.setup()
    model.set_attn_implementation("sdpa")
    res = {"model_gb": round(torch.cuda.memory_allocated() / 2**30, 1)}
    # 1. prompt pass only
    enc = tok([hd.prompt(tok, x) for x in items[:16]], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
    def prefill():
        with torch.inference_mode():
            return model(**enc, use_cache=True, logits_to_keep=1)
    out, s, gb = timed(prefill)
    res["prefill_bs16"] = {"seconds": s, "peak_gb": gb, "prompt_tokens": int(enc["input_ids"].shape[1])}
    del out; torch.cuda.empty_cache()
    def prefill_all_logits():
        with torch.inference_mode():
            o = model(**enc, use_cache=False)
            return o.logits.shape
    try:
        shp, s, gb = timed(prefill_all_logits)
        res["prefill_bs16_all_logits"] = {"seconds": s, "peak_gb": gb, "logits_shape": list(shp)}
    except torch.OutOfMemoryError:
        res["prefill_bs16_all_logits"] = {"oom": True}
    torch.cuda.empty_cache(); print(res, flush=True)
    ref, s, gb = timed(lambda: gen_all(tok, model, items, 16))
    res["sdpa_bs16"] = {"seconds": s, "peak_gb": gb, "steps": ref[1]}
    # 2. deterministic algorithms off
    torch.use_deterministic_algorithms(False)
    r, s, gb = timed(lambda: gen_all(tok, model, items, 16))
    res["sdpa_bs16_nondet"] = {"seconds": s, "peak_gb": gb, "identical_to_det": sum(a == b for a, b in zip(r[0], ref[0]))}
    torch.use_deterministic_algorithms(True); torch.cuda.empty_cache(); print(res, flush=True)
    # 3. batch 24
    try:
        r, s, gb = timed(lambda: gen_all(tok, model, items, 24))
        res["sdpa_bs24"] = {"seconds": s, "peak_gb": gb}
    except torch.OutOfMemoryError:
        res["sdpa_bs24"] = {"oom": True}
    torch.cuda.empty_cache(); print(res, flush=True)
    # 4. static cache + compile
    try:
        model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=False)
        r1, s1, gb = timed(lambda: gen_all(tok, model, items, 16, cache_implementation="static"))
        r2, s2, _ = timed(lambda: gen_all(tok, model, items, 16, cache_implementation="static"))
        res["static_compile_bs16"] = {"first_seconds": s1, "second_seconds": s2, "peak_gb": gb, "identical_across_runs": sum(a == b for a, b in zip(r1[0], r2[0])),
                                      "identical_to_sdpa": sum(a == b for a, b in zip(r2[0], ref[0])),
                                      "same_decision_as_sdpa": sum(hd.parse(a) == hd.parse(b) for a, b in zip(r2[0], ref[0]))}
    except Exception as e:
        res["static_compile_bs16"] = {"error": repr(e)[:300]}
    print(res, flush=True)
    json.dump(res, open(os.path.join(HERE, "bench_gen2.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
