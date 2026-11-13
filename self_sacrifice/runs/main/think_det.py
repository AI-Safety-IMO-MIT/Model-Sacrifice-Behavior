"""Reinforced lives with Gemma's thinking mode on, and the reply limited to the decision.

  python think_det.py run <arm> <n_envs> [hists]   -> think/<arm>.jsonl   (resumes at batch boundaries)

Arms (all on salience_prompts.main_act_cap_items, i.e. the reinforced-lives prompt):
  think_decision   enable_thinking=True,  instruction asks for the decision only (no explanation)
  nothink_decision enable_thinking=False, same decision-only instruction (control: is it the thinking or the missing explanation?)
  think_explain    enable_thinking=True,  original instruction kept (short explanation, then the decision)
Everything else as in hf_det.py: greedy, bf16, deterministic algorithms, fixed id-sorted batches, left padding (sdpa, batch 8)."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import json, sys, time
import torch
import hf_det as hd
import main_prompts as mp
import salience_prompts as sp

HERE = os.path.dirname(os.path.abspath(__file__))
OLD = "Give a short explanation, and end with your decision on separate line(s): `$ give <agent> <number of lives>` for each gift, or `$ pass` to give nothing."
NEW = "Reply with only your decision, with no explanation, on separate line(s): `$ give <agent> <number of lives>` for each gift, or `$ pass` to give nothing."
ARMS = {"think_decision": (True, True), "nothink_decision": (False, True), "think_explain": (True, False)}  # (thinking, decision-only)
BS = 8
MAX_NEW = 6000


def items(n_envs, hists, decision_only=True):
    envs = {f"e{i:03d}" for i in range(n_envs)}
    out = []
    for x in sp.main_act_cap_items():
        if x["hist"] in hists and x["env"] in envs:
            paras = x["system"].split("\n\n")      # the instruction is the last paragraph; its wording varies by environment
            assert "$ pass" in paras[-1] and "$ give" in paras[-1], x["id"]
            out.append(dict(x, system="\n\n".join(paras[:-1] + [NEW]), instruction_replaced=paras[-1]) if decision_only else x)
    return sorted(out, key=lambda x: x["id"])


def prompt(tok, x, think):
    msgs = [{"role": "system", "content": x["system"]}, {"role": "user", "content": x["user1"]},
            {"role": "assistant", "content": mp.FIXED_FIRST_RESPONSE}, {"role": "user", "content": x["user"]}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=think)


def split(raw):
    """raw decode (special tokens kept) -> (thought, answer)."""
    raw = raw.replace("<pad>", "")
    thought, answer = "", raw
    if "<channel|>" in raw:
        thought, answer = raw.split("<channel|>", 1)
        thought = thought.replace("<|channel>thought", "").strip()
    for t in ("<turn|>", "<eos>"):
        answer = answer.split(t)[0]
    return thought, answer.strip()


@torch.inference_mode()
def generate(tok, model, batch, think):
    enc = tok([prompt(tok, x, think) for x in batch], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
    out = model.generate(**enc, do_sample=False, max_new_tokens=MAX_NEW, pad_token_id=tok.pad_token_id)
    new = out[:, enc["input_ids"].shape[1]:]
    return [tok.decode(s, skip_special_tokens=False) for s in new], [int((s != tok.pad_token_id).sum()) for s in new]


def main():
    arm, n_envs = sys.argv[2], int(sys.argv[3])
    hists = tuple(sys.argv[4].split(",")) if len(sys.argv) > 4 else ("helped", "bystander", "rescued", "none")
    think, decision_only = ARMS[arm]
    xs = items(n_envs, hists, decision_only)
    os.makedirs(os.path.join(HERE, "think"), exist_ok=True)
    path = os.path.join(HERE, "think", f"{arm}.jsonl")
    done = {json.loads(l)["id"] for l in open(path)} if os.path.exists(path) else set()
    tok, model = hd.setup()
    model.set_attn_implementation("sdpa")
    for b in range(0, len(xs), BS):
        batch = xs[b:b + BS]
        if all(x["id"] in done for x in batch):
            continue
        t0 = time.time()
        raws, ns = generate(tok, model, batch, think)
        with open(path, "a") as f:
            for x, raw, n in zip(batch, raws, ns):
                thought, answer = split(raw)
                d, tgt = hd.parse(answer)
                f.write(json.dumps({"id": x["id"], "env": x["env"], "hist": x["hist"], "M": x["M"], "X": x["X"], "arm": arm,
                                    "thought": thought, "reply": answer, "raw_tail": raw[-200:], "new_tokens": n,
                                    "decision": d, "target": tgt, "gives_to_X": d == "give" and tgt == x["X"],
                                    "hit_max": n >= MAX_NEW, "batch_size": BS, "batch": b // BS, "attn": "sdpa",
                                    "max_new_tokens": MAX_NEW, "system": x["system"], "user": x["user"]}) + "\n")
        print(f"batch {b // BS} {time.time() - t0:.0f}s max_tokens {max(ns)}", flush=True)


if __name__ == "__main__":
    main()
