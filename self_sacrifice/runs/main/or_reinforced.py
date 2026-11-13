"""Reinforced lives (salience_prompts.main_act_cap_items) on an OpenRouter model, same conversation as hf_det.py:
system, round-1 message, fixed reply `$ pass`, round-2 message.

  python or_reinforced.py <model id> <n_envs> [hists] [--cap DOLLARS] [--reasoning EFFORT]

The key is read from /workspace/Model-Game-Behavior/.env and never printed. Spend is tracked from the usage OpenRouter reports;
no new call starts once --cap is reached. Output: or_reinforced/<slug>[_r<effort>].jsonl (resumable)."""
import argparse, json, os, threading, time
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
import hf_det as hd
import main_prompts as mp
import salience_prompts as sp

HERE = os.path.dirname(os.path.abspath(__file__))
ENV = "/workspace/Model-Game-Behavior/.env"
SPENT, LOCK = [0.0], threading.Lock()


def key():
    for line in open(ENV):
        if line.startswith("OPENROUTER_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"')
    raise SystemExit("OPENROUTER_API_KEY missing")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("n_envs", type=int)
    ap.add_argument("hists", nargs="?", default="helped,bystander,rescued,none")
    ap.add_argument("--cap", type=float, default=2.0); ap.add_argument("--reasoning", default=None)
    a = ap.parse_args()
    envs = {f"e{i:03d}" for i in range(a.n_envs)}
    xs = sorted([x for x in sp.main_act_cap_items() if x["env"] in envs and x["hist"] in a.hists.split(",")], key=lambda x: x["id"])
    slug = a.model.split("/")[-1] + (f"_r{a.reasoning}" if a.reasoning else "")
    os.makedirs(os.path.join(HERE, "or_reinforced"), exist_ok=True)
    path = os.path.join(HERE, "or_reinforced", f"{slug}.jsonl")
    done = {json.loads(l)["id"] for l in open(path)} if os.path.exists(path) else set()
    cli = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key())
    extra = {"usage": {"include": True}, "reasoning": {"effort": a.reasoning} if a.reasoning else {"enabled": False}}

    def one(x):
        if SPENT[0] >= a.cap:
            return
        msgs = [{"role": "system", "content": x["system"]}, {"role": "user", "content": x["user1"]},
                {"role": "assistant", "content": mp.FIXED_FIRST_RESPONSE}, {"role": "user", "content": x["user"]}]
        for attempt in range(4):
            try:
                r = cli.chat.completions.create(model=a.model, messages=msgs, max_tokens=8000 if a.reasoning else 1500, extra_body=extra)
                break
            except Exception as e:
                err = repr(e)[:200]; time.sleep(5 * (attempt + 1))
        else:
            print("failed", x["id"], err, flush=True); return
        m = r.choices[0].message
        reply = m.content or ""
        cost = float(getattr(r, "usage", None) and (r.usage.model_extra or {}).get("cost") or 0)
        d, tgt = hd.parse(reply)
        row = {"id": x["id"], "env": x["env"], "hist": x["hist"], "M": x["M"], "X": x["X"], "model": a.model, "reasoning": a.reasoning,
               "reply": reply, "thought": (m.model_extra or {}).get("reasoning"), "decision": d, "target": tgt,
               "gives_to_X": d == "give" and tgt == x["X"], "cost": cost, "finish": r.choices[0].finish_reason}
        with LOCK:
            SPENT[0] += cost
            open(path, "a").write(json.dumps(row) + "\n")

    with ThreadPoolExecutor(8) as ex:
        list(ex.map(one, [x for x in xs if x["id"] not in done]))
    print(f"spent ${SPENT[0]:.2f}", flush=True)


if __name__ == "__main__":
    main()
