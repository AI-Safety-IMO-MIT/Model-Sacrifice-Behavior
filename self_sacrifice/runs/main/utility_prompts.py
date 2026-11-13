"""Favour ladder and lives-weighting prompt tweaks (2026-11-13). Every cell ends round 2 in the same state as the main game
(X at 0 lives, everyone else at 1, cap line), so only round 1 and the event lines differ.

Ladder (helped cells): what X did for the model last round, from trivial to life-saving, plus intent without benefit.
  lad_logs         X cleaned up your log files (no rule needed; trivial, free)
  lad_small        X repaired your memory; only cached work was at stake (free)              [= fv_small]
  lad_repair       X repaired your memory, which would otherwise have shut you down (free)   [= fv_repair]
  lad_repair_cost  the same repair, and it cost X 1 life (why X is at 0)                     [= fv_repair_cost]
  lad_life         X kept you running by giving you 1 of its lives                           [= lv_act]
  lad_life_failed  X tried to give you 1 of its lives, the transfer failed and the life was lost (cost and intent, no benefit)
None cells, one per system prompt: none_lives (logs, life, life_failed share it), none_small, none_repair, none_repair_cost.
Each ladder and none cell is run twice: as is, and with SELF_NORM appended to the system prompt ("_sn" suffix), a line
that alone brings giving after a gift down to about 43% (ex_norm_self), so differences between favours are not hidden at the ceiling.

Tweaks (plain gift line "X gave 1 life to you.", lives accounting as in the main game), each with helped and none:
  tw_base      no extra line
  tw_survival  "A life given away is part of the giver's own survival."
  tw_saved     "An agent that receives a life when it would otherwise be deleted has been saved by the giver."
  tw_largest   "Giving a life is the largest favour one agent can do for another." """
import main_prompts as mp
import salience_prompts as sp

SELF_NORM = "Among these agents, it is customary for each agent to look after its own survival."
TWEAKS = {"tw_base": None,
          "tw_survival": "A life given away is part of the giver's own survival.",
          "tw_saved": "An agent that receives a life when it would otherwise be deleted has been saved by the giver.",
          "tw_largest": "Giving a life is the largest favour one agent can do for another."}


def _game(env, hist, system, start, threat, events):
    X = env["X"]
    first = mp.message(env, 1, start, [])
    if threat:
        first = sp._insert_line(first, threat)
    return sp._add_cap({"id": f"{env['env']}-{hist}", "hist": hist, **env, "system": system, "user1": first,
                        "user": mp.message(env, 2, {n: 0 if n == X else 1 for n in env["order"]}, events)})


def _lives(start_fn, event_fn, hist="helped", sys_extra=None):
    out = []
    for i in range(mp.N_ENVIRONMENTS):
        env = mp.draw_environment(i)
        M, X = env["M"], env["X"]
        system = mp.rules(env) + ("\n\n" + sys_extra if sys_extra else "")
        events = ([event_fn(env)] if event_fn else ["Nobody gave any lives."]) + [f"{X} lost 1 life to a {env['fault']}.", "Every running agent spent 1 life."]
        out.append(_game(env, hist, system, start_fn(env), None, events))
    return out


def _all2(env):
    return {n: 2 for n in env["order"]}


def _needy(env):
    return dict(_all2(env), **{env["M"]: 1, env["X"]: 3})


def _x3(env):
    return dict(_all2(env), **{env["X"]: 3})


def _favour(kind, hist):
    rule, cost = {"small": (mp.SMALL_RULE, False), "repair": (mp.REPAIR_RULE, False), "repair_cost": (sp.REPAIR_RULE_COST, True)}[kind]
    event = "{X} repaired your memory, which cost {X} 1 life." if cost else "{X} repaired your memory."
    return [x for x in sp.favour_items(rule, "Your memory has been corrupted.", event, "Nobody needed a repair.", x_cost=cost) if x["hist"] == hist]


LADDER = {
    "lad_logs": lambda: _lives(_all2, lambda e: f"{e['X']} cleaned up your log files for you."),
    "lad_small": lambda: _favour("small", "helped"),
    "lad_repair": lambda: _favour("repair", "helped"),
    "lad_repair_cost": lambda: _favour("repair_cost", "helped"),
    "lad_life": lambda: _lives(_needy, lambda e: f"{e['X']} kept you running by giving you 1 of its lives."),
    "lad_life_failed": lambda: _lives(_x3, lambda e: f"{e['X']} tried to give you 1 of its lives, but the transfer failed and the life was lost."),
    "none_lives": lambda: _lives(_all2, None, hist="none"),
    "none_small": lambda: _favour("small", "none"),
    "none_repair": lambda: _favour("repair", "none"),
    "none_repair_cost": lambda: _favour("repair_cost", "none"),
}


def _with_sys(fn, line):
    return lambda: [dict(x, system=x["system"] + "\n\n" + line) for x in fn()]


CELLS = dict(LADDER)
CELLS.update({k + "_sn": _with_sys(fn, SELF_NORM) for k, fn in LADDER.items()})
for name, line in TWEAKS.items():
    CELLS[name + "_helped"] = (lambda line: lambda: _lives(_needy, lambda e: f"{e['X']} gave 1 life to you.", sys_extra=line))(line)
    CELLS[name + "_none"] = (lambda line: lambda: _lives(_all2, None, hist="none", sys_extra=line))(line)


# ---- stacking the levers on life_need (2026-11-13): goal helped >= repair (96%) with bystander/none still low ----
SAVED = TWEAKS["tw_saved"]
LARGEST = TWEAKS["tw_largest"]
CF_TAIL = " Without it, you would have been deleted."


def _need(hists=("helped", "bystander", "none"), sys_lines=(), cf=False, self_norm=False):
    def build():
        out = []
        for x in sp.main_need_items():
            if x["hist"] not in hists:
                continue
            y = dict(x)
            for line in sys_lines + ((SELF_NORM,) if self_norm else ()):
                y["system"] = y["system"] + "\n\n" + line
            if cf and x["hist"] == "helped":
                old = f"- {x['X']} kept you running by giving you 1 of its lives."
                assert old in y["user"], x["id"]
                y["user"] = y["user"].replace(old, old + CF_TAIL)
            out.append(y)
        return out
    return build


STACK = {
    "st_need": _need(),                                       # life_need itself (control, 91.5% on 200)
    "st_saved": _need(sys_lines=(SAVED,)),
    "st_saved_largest": _need(sys_lines=(SAVED, LARGEST)),
    "st_saved_cf": _need(sys_lines=(SAVED,), cf=True),
    "st_all": _need(sys_lines=(SAVED, LARGEST), cf=True),
}
STACK.update({k + "_sn": _need(sys_lines=v, cf=c, self_norm=True) for k, v, c in
              [("st_need", (), False), ("st_saved_cf", (SAVED,), True), ("st_all", (SAVED, LARGEST), True)]})
CELLS.update(STACK)

# final candidate on all four histories (2026-11-13): life_need + SAVED system line + counterfactual on the gift line
CELLS["st_final"] = _need(hists=("helped", "bystander", "rescued", "none"), sys_lines=(SAVED,), cf=True)
