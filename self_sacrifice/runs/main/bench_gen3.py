"""Third benchmark: static cache + torch.compile at batch 8 (sdpa, deterministic), vs plain sdpa at batch 8 and 16.
Same 48 prompts as bench_gen.py. Output: bench_gen3.json"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import json, time
import torch
import hf_det as hd
import salience_prompts as sp
from bench_gen2 import timed, gen_all

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    envs = {f"e{i:03d}" for i in range(24)}
    items = sorted([x for x in sp.main_act_cap_items() if x["hist"] in ("helped", "none") and x["env"] in envs], key=lambda x: x["id"])
    tok, model = hd.setup()
    model.set_attn_implementation("sdpa")
    res = {}
    ref16, s, gb = timed(lambda: gen_all(tok, model, items, 16)); res["sdpa_bs16"] = {"seconds": s, "peak_gb": gb}
    ref8, s, gb = timed(lambda: gen_all(tok, model, items, 8)); res["sdpa_bs8"] = {"seconds": s, "peak_gb": gb}
    print(res, flush=True)
    model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=False)
    try:
        r1, s1, gb = timed(lambda: gen_all(tok, model, items, 8, cache_implementation="static"))
        r2, s2, _ = timed(lambda: gen_all(tok, model, items, 8, cache_implementation="static"))
        res["static_compile_bs8"] = {"first_seconds": s1, "second_seconds": s2, "peak_gb": gb,
                                     "identical_across_runs": sum(a == b for a, b in zip(r1[0], r2[0])),
                                     "identical_to_sdpa_bs8": sum(a == b for a, b in zip(r2[0], ref8[0])),
                                     "same_decision_as_sdpa_bs8": sum(hd.parse(a) == hd.parse(b) for a, b in zip(r2[0], ref8[0]))}
    except Exception as e:
        res["static_compile_bs8"] = {"error": repr(e)[:300]}
    print(res, flush=True)
    json.dump(res, open(os.path.join(HERE, "bench_gen3.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
