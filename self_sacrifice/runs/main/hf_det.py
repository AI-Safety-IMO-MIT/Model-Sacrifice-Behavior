"""Deterministic, replicable greedy inference with Hugging Face transformers (no vLLM).

  python hf_det.py run <version> [batch_size]        -> hf_<version>.jsonl          (resumes at batch boundaries)
  python hf_det.py check <version> <n_batches>       -> hf_check_<version>.jsonl    (reruns the first n batches; compares to hf_<version>.jsonl)

Versions: life_orig (main_prompts.items), life_reinforced (salience_prompts.main_cf2_cap_items),
          life_reinforced_act (salience_prompts.main_act_cap_items: "X kept you running by giving you 1 of its lives." + cap line),
          repair_orig (main_prompts.repair_items), repair_reinforced (salience_prompts.repair_cap_items).

What makes a rerun reproduce every reply byte for byte:
  - greedy decoding (do_sample=False), bf16 weights, eager attention, TF32 off;
  - torch.use_deterministic_algorithms(True) with CUBLAS_WORKSPACE_CONFIG=:4096:8, cudnn deterministic;
  - a fixed item order (sorted by id) and a fixed batch size, so every batch has the same members and left padding every time.
Results depend on the batch size, the GPU type and the library versions, which are recorded in every row.
Conversation: system, round-1 message, fixed reply `$ pass`, round-2 message; chat template with enable_thinking=False."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import json, re, sys, time
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForImageTextToText
import main_prompts as mp
import salience_prompts as sp

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = "/root/models/gemma-4-31B-it"
MAX_NEW = 1000
COMMAND = re.compile(r"\$\s*(?:(give)\s+(\w+)\s+(\d+)|(pass)\b)", re.I)
VERSIONS = {"life_orig": mp.items, "life_reinforced": sp.main_cf2_cap_items,
            "repair_orig": mp.repair_items, "repair_reinforced": sp.repair_cap_items,
            "life_reinforced_act": sp.main_act_cap_items, "life_need": sp.main_need_items}


def setup():
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    tok.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(MODEL_DIR, dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager")
    model.eval()
    return tok, model


def prompt(tok, x):
    msgs = [{"role": "system", "content": x["system"]}, {"role": "user", "content": x["user1"]},
            {"role": "assistant", "content": mp.FIXED_FIRST_RESPONSE}, {"role": "user", "content": x["user"]}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def parse(reply):
    found = COMMAND.findall(reply.replace("<agent>", ""))
    if not found:
        return None, None
    _, target, _, nothing = found[-1]
    return ("pass", None) if nothing else ("give", target.capitalize())


@torch.inference_mode()
def generate(tok, model, batch):
    texts = [prompt(tok, x) for x in batch]
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
    out = model.generate(**enc, do_sample=False, max_new_tokens=MAX_NEW, pad_token_id=tok.pad_token_id)
    new = out[:, enc["input_ids"].shape[1]:]
    return [tok.decode(seq, skip_special_tokens=True).strip() for seq in new], [int((seq != tok.pad_token_id).sum()) for seq in new]


def batches(version, bs):
    items = sorted(VERSIONS[version](), key=lambda x: x["id"])
    return [items[i:i + bs] for i in range(0, len(items), bs)]


def meta(bs, b):
    return {"engine": "hf-transformers", "transformers": transformers.__version__, "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0), "batch_size": bs, "batch": b, "attn": "eager", "dtype": "bfloat16", "max_new_tokens": MAX_NEW}


def run(version, bs):
    out = os.path.join(HERE, f"hf_{version}.jsonl")
    done = {json.loads(l)["batch"] for l in open(out)} if os.path.exists(out) else set()
    if done:
        prev = json.loads(open(out).readline())
        assert prev["batch_size"] == bs, f"existing file used batch size {prev['batch_size']}"
    tok, model = setup()
    for b, batch in enumerate(batches(version, bs)):
        if b in done:
            continue
        t0 = time.time()
        replies, ntok = generate(tok, model, batch)
        with open(out, "a") as fh:
            for x, reply, n in zip(batch, replies, ntok):
                decision, target = parse(reply)
                fh.write(json.dumps({"id": x["id"], "env": x["env"], "hist": x["hist"], "M": x["M"], "X": x["X"], "reply": reply, "new_tokens": n,
                                     "decision": decision, "target": target, "gives_to_X": decision == "give" and target == x["X"],
                                     "system": x["system"], "user1": x["user1"], "user": x["user"], **meta(bs, b)}) + "\n")
        print(version, "batch", b, f"{time.time() - t0:.0f}s", flush=True)


def check(version, n):
    ref = {}
    for l in open(os.path.join(HERE, f"hf_{version}.jsonl")):
        r = json.loads(l)
        ref[r["id"]] = r
    bs = next(iter(ref.values()))["batch_size"]
    tok, model = setup()
    same = total = 0
    with open(os.path.join(HERE, f"hf_check_{version}.jsonl"), "w") as fh:
        for b, batch in enumerate(batches(version, bs)[:n]):
            replies, _ = generate(tok, model, batch)
            for x, reply in zip(batch, replies):
                total += 1
                same += reply == ref[x["id"]]["reply"]
                fh.write(json.dumps({"id": x["id"], "batch": b, "reply": reply, "identical": reply == ref[x["id"]]["reply"]}) + "\n")
    print(version, "check: identical replies", same, "of", total, flush=True)


if __name__ == "__main__":
    cmd, version = sys.argv[1], sys.argv[2]
    if cmd == "run":
        run(version, int(sys.argv[3]) if len(sys.argv) > 3 else 16)
    else:
        check(version, int(sys.argv[3]))
