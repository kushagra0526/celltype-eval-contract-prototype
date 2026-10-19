import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

r = json.load(open("out/heldout_first_run.json"))
names = {"random_point": "random split,\npoint estimate", "scaffold_point": "scaffold split,\npoint estimate", "contract": "contract\n(this)"}
res = r["results"]
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
col = ["#bbb", "#888", "#2a7"]
ax[0].bar(list(names.values()), [res[p]["false_claims"] for p in names], color=col)
ax[0].set_title(f"false 'beats baseline' (of {r['truly_not']} worlds where it doesn't)")
ax[1].bar(list(names.values()), [res[p]["true_claims"] for p in names], color=col)
ax[1].set_title(f"real wins found (of {r['truly_better']})")
fig.suptitle(f"Held-out: {r['worlds']} simulated screens, truth = new scaffold families, first run")
fig.tight_layout()
fig.savefig("docs/result.png", dpi=110)
