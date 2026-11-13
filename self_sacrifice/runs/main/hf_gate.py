import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
ok = sum(r["identical"] for r in rows)
print("gate:", ok, "of", len(rows), "identical", flush=True)
sys.exit(0 if rows and ok == len(rows) else 1)
