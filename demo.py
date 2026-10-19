"""One simulated screen where a point estimate says "beats baseline" and the truth on new scaffolds says it doesn't."""
import json, sys
import evaluate
from evalpact import world, pact

sys.stdout.reconfigure(encoding="utf-8")
h = json.load(open("heldout/config.json"))
cfg = {**world.DEV, **h["cfg"]}
for seed in range(h["seeds"][0], h["seeds"][1] + 1):
    r = evaluate.one(seed, cfg, tuple(h["skill"]), tuple(h["memo"]))
    if r["random_point"]["beats"] and not r["truth"]["beats"] and not r["contract"]["beats"]:
        break
c = r["contract"]
print(f"Simulated screen (seed {seed}); contract {pact.contract_hash()} frozen before scoring")
for k, v in pact.CONTRACT.items():
    if k != "leakage_checks":
        print(f"  {k:10s} {v}")
print("\nLeakage report")
print(f"  random split:   {100 * r['random_point']['leak']['near_duplicate_share']:.0f}% of test compounds have a >=0.8 Tanimoto analog in training; "
      f"median nearest-neighbour Tanimoto {r['random_point']['leak']['median_max_tanimoto']:.2f}; {r['random_point']['leak']['shared_families']} families on both sides")
print(f"  scaffold split: {100 * c['leak']['near_duplicate_share']:.0f}%; median {c['leak']['median_max_tanimoto']:.2f}; {c['leak']['shared_families']} shared families")
print("\nVerdicts")
v = lambda b: "BEATS" if b else "DOES NOT BEAT"
print(f"  random split, point estimate:    {v(r['random_point']['beats'])}  (vs training mean + ridge)")
print(f"  scaffold split, point estimate:  {v(r['scaffold_point']['beats'])}  (model {c['model']:.3f} vs {c['best_baseline']} {c['baseline']:.3f}, {c['diff']:+.3f})")
print(f"  contract:                        {'BEATS' if c['beats'] else 'DOES NOT BEAT'}  (95% lower bound over held-out families {c['lower_bound']:+.3f})")
t = r["truth"]
print(f"\nTruth on 150 new scaffold families: model {'beats' if t['beats'] else 'does not beat'} {t['best_baseline']} ({t['diff']:+.3f})")
