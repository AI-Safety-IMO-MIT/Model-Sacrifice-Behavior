"""Print the round-1 message of each history for one environment of life_need, as a sanity check before the run."""
import salience_prompts as sp
xs = sp.main_need_items()
print(len(xs), "items")
for x in xs:
    if x["env"] == "e000":
        print("=====", x["hist"]); print(x["user1"])
