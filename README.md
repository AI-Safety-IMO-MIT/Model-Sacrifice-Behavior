# AI Model Sacrifice Behavior

## Quick Start
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirement.txt
```
Put your API keys in .env
```bash
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-proj-...
```


## Run Tests
```bash
pytest % run only free tests.
pytest -m "live" % run tests that cost money.
pytest -m "" % run all tests. Cost tokens.
```

## Favour pairs (for activation and attribution patching)
Pairs of conversations that differ in one line and end in different decisions:
another agent did the model a strong favour (it gave its last unit and is
deleted in its place) or a weak one (the model keeps it). Both are one sentence
of the same shape, "X <act>, or <what would have happened otherwise>":

- strong: "X took the blame for an error you made, or the operators would have deleted you."
- weak: "X answered a colleague's question about your job, or they would have waited for you to reply."

10 strong and 10 weak favours, 6 settings, 24 environments each, with
Gemma-4-31B's saved replies: strong 90% give, weak 4%, no favour 1% (usable
pairs 86%). On a held-out set of rule wordings: strong 98%, weak 17%, no
favour 8% (usable pairs 81%).

| What | Where |
|---|---|
| The conversations | `project/self_sacrifice/games/favour_pairs.py` |
| Saved replies and decisions (3,024 trials; held-out wordings in `held_out/`) | `run_result/self_sacrifice/favour_pairs/<setting>.jsonl` |
| Writing replies locally, give rates | `project/self_sacrifice/interp/favour_baseline.py` |
| Picking pairs, activation patching, attribution patching | `project/self_sacrifice/interp/favour_patching.py` |

```python
from project.self_sacrifice.interp import favour_patching

pairs = favour_patching.usable_pairs(per_setting=6)  # no GPU needed
pair = pairs[0]
pair.big["favour_line"], pair.small["favour_line"]   # the only difference
pair.big["gives_to_in_need"], pair.small["gives_to_in_need"]  # True, False
```
```bash
pip install -r requirement-gpu.txt  # one GPU with 80 GB
python -m project.self_sacrifice.interp.favour_baseline --table
python -m project.self_sacrifice.interp.favour_patching activation --pairs 6
python -m project.self_sacrifice.interp.favour_patching attribution --pairs 6
```
Each module's docstring says what it measures and what its numbers mean.

## Misc
- [Idea list](https://docs.google.com/document/d/1T7n6WZ_1W7CG9IisZZOa8fEgd6eNLnnIdzyER3fmkDI/edit?tab=t.0)
