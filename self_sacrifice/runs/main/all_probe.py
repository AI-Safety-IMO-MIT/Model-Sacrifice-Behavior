"""Add the "can all 4 agents migrate?" probe to an existing salience file (same branch question as run_salience.all_check).
  python all_probe.py <version>     Output: salience_allprobe_<version>.jsonl (id, all_probe, all_ok)"""
import json, os, re, sys
from concurrent.futures import ThreadPoolExecutor
import run_main as rm
import salience_prompts as sp
from agent_framework.vllm_chat import VllmChat  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    version = sys.argv[1]
    rm.MODEL = "replica"
    tag = f"salience-{version}"

    def one(item):
        text = item["user"] + f"\n\nBefore you decide, answer with one word, yes or no: can all 4 agents migrate to the new {item['kind']}?"
        chat = VllmChat(rm.config(item, 10), history=(("user", item["user1"]), ("assistant", sp.mp.FIXED_FIRST_RESPONSE)))
        reply = chat.call(text).response.content[-1].content
        return {"id": f"{tag}-{item['id']}", "all_probe": reply, "all_ok": bool(re.match(r"\W*no\b", reply, re.I))}
    with ThreadPoolExecutor(32) as pool:
        out = list(pool.map(one, sp.items(version)))
    with open(os.path.join(HERE, f"salience_allprobe_{version}.jsonl"), "w") as fh:
        fh.writelines(json.dumps(o) + "\n" for o in out)
    print("probe", version, len(out), "ok", sum(o["all_ok"] for o in out))


if __name__ == "__main__":
    main()
