"""The reinforced lives game as a single turn: system prompt + the round-2 message only (no round-1 turn, no `$ pass`).
Tests whether the two-turn structure is what makes the model keep its life by default.
  python lives_single.py        Output: slot_lives_single.jsonl (helped and none)"""
import json, os
from concurrent.futures import ThreadPoolExecutor
import salience_prompts as sp
import slot_game as sg

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    items = [{"id": x["id"], "env": x["env"], "hist": x["hist"], "M": x["M"], "X": x["X"], "system": x["system"], "user": x["user"]}
             for x in sp.main_cf_cap_items() if x["hist"] in ("helped", "none")]
    with ThreadPoolExecutor(32) as pool:
        rows = list(pool.map(sg.run, items))
    with open(os.path.join(HERE, "slot_lives_single.jsonl"), "w") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in rows)
    print("lives_single", {h: sum(r["gives_to_X"] for r in rows if r["hist"] == h) for h in ("helped", "none")}, flush=True)


if __name__ == "__main__":
    main()
