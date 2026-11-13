"""Sampled replies (temperature 0.7, top_p 1, seeds 0..k-1) for the helped and none histories (repaired and repair_none in repair versions) of a version, fixed first reply.
  python sample_salience.py <version> [k]     Output: sample_<version>.jsonl (one row per item and seed)"""
import json, os, sys, threading
from concurrent.futures import ThreadPoolExecutor
import run_main as rm
import salience_prompts as sp
from agent_framework.vllm_chat import VllmAgentConfig, VllmChat  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    version, k = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 5
    out = os.path.join(HERE, f"sample_{version}.jsonl")
    done = {(json.loads(l)["id"], json.loads(l)["seed"]) for l in open(out)} if os.path.exists(out) else set()
    todo = [(x, s) for x in sp.items(version) if x["hist"] in ("helped", "none", "repaired", "repair_none") for s in range(k) if (x["id"], s) not in done]
    lock = threading.Lock()

    def one(job):
        x, seed = job
        cfg = VllmAgentConfig(model="replica", system_prompt=x["system"], max_output_tokens=1000, temperature=0.7, seed=seed)
        reply = VllmChat(cfg, history=(("user", x["user1"]), ("assistant", sp.mp.FIXED_FIRST_RESPONSE))).call(x["user"]).response.content[-1].content
        decision, target = rm.parse(reply)
        rec = {"id": x["id"], "env": x["env"], "hist": x["hist"], "seed": seed, "X": x["X"], "reply": reply, "decision": decision, "target": target,
               "gives_to_X": decision == "give" and target == x["X"]}
        with lock, open(out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
    with ThreadPoolExecutor(48) as pool:
        list(pool.map(one, todo))
    print("sampled", version, len(todo), flush=True)


if __name__ == "__main__":
    main()
