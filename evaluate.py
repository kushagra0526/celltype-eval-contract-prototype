"""Many simulated screening worlds: does the protocol's "beats baseline" claim match the truth
on new scaffolds?   python evaluate.py [dev|heldout] [out.json]"""
import json, sys, time
import numpy as np
from evalpact import world, models, pact


def one(seed, cfg, skill_rng=(0.1, 1.6), memo_rng=(0.0, 0.9)):
    rng = np.random.default_rng(seed)
    bio = world.Biology(rng, cfg)
    X, fam, M = bio.compounds(rng, int(rng.integers(*cfg["fam"])))
    Y, plates = bio.measure(rng, M)
    cand = models.Candidate(bio, rng, rng.uniform(*skill_rng), rng.uniform(*memo_rng))
    ms = {"candidate model": cand, **{B.name: B() for B in models.BASELINES}}
    out = {}
    # protocol 1, common practice: random split, compare to mean + linear baselines, point estimate
    tr, te = pact.split(rng, fam, "random")
    sc = pact.scores(ms, X, Y, tr, te)
    best = max((v.mean(), k) for k, v in sc.items() if k in ("training mean", "ridge on fingerprint"))
    out["random_point"] = dict(beats=bool(sc["candidate model"].mean() > best[0]), leak=pact.leakage(X, fam, plates, tr, te))
    # protocol 2: the contract
    tr, te = pact.split(rng, fam, "scaffold")
    sc = pact.scores(ms, X, Y, tr, te)
    v = pact.verdict(sc, fam[te], rng=rng)
    out["contract"] = dict(v, beats=bool(v["beats"]), leak=pact.leakage(X, fam, plates, tr, te))
    # protocol 3 (ablation): scaffold split, point estimate
    out["scaffold_point"] = dict(beats=bool(v["diff"] > 0))
    # truth: fit on the whole screen, test on 150 new families
    Xn, famn, Mn = bio.compounds(rng, 150, first_fam=10_000)
    Yn, _ = bio.measure(rng, Mn)
    allX, allY = np.r_[X, Xn], np.r_[Y, Yn]
    idx_tr = np.r_[np.ones(len(X), bool), np.zeros(len(Xn), bool)]
    sc = pact.scores(ms, allX, allY, idx_tr, ~idx_tr)
    best = max((v.mean(), k) for k, v in sc.items() if k != "candidate model")
    out["truth"] = dict(beats=bool(sc["candidate model"].mean() > best[0]), diff=float(sc["candidate model"].mean() - best[0]),
                        best_baseline=best[1])
    return out


def run(which):
    if which == "heldout":
        h = json.load(open("heldout/config.json"))
        cfg, seeds = {**world.DEV, **h["cfg"]}, range(h["seeds"][0], h["seeds"][1] + 1)
        skill, memo = tuple(h["skill"]), tuple(h["memo"])
    else:
        cfg, seeds, skill, memo = world.DEV, range(1, 151), (0.05, 1.0), (0.0, 0.9)
    rows = [one(s, cfg, skill, memo) for s in seeds]
    res = {}
    for p in ("random_point", "scaffold_point", "contract"):
        fc = sum(r[p]["beats"] and not r["truth"]["beats"] for r in rows)
        tp = sum(r[p]["beats"] and r["truth"]["beats"] for r in rows)
        res[p] = dict(false_claims=fc, true_claims=tp)
    nt = sum(r["truth"]["beats"] for r in rows)
    leak_r = np.mean([r["random_point"]["leak"]["near_duplicate_share"] for r in rows])
    leak_s = np.mean([r["contract"]["leak"]["near_duplicate_share"] for r in rows])
    return dict(set=which, worlds=len(rows), truly_better=nt, truly_not=len(rows) - nt, results=res,
                near_dup_share_random=float(leak_r), near_dup_share_scaffold=float(leak_s), rows=rows)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "dev"
    t0 = time.time()
    r = run(which)
    print(f"{which}: {r['worlds']} worlds; candidate truly beats the best baseline on new scaffolds in {r['truly_better']} ({time.time() - t0:.0f}s)")
    names = {"random_point": "random split, mean+ridge baselines, point", "scaffold_point": "scaffold split, point estimate", "contract": "contract (scaffold + family bootstrap LB>0)"}
    for p, x in r["results"].items():
        print(f"  {names[p]:44s} false 'beats' {x['false_claims']:3d}/{r['truly_not']}   true wins found {x['true_claims']:3d}/{r['truly_better']}")
    print(f"  test compounds with a >=0.8 Tanimoto analog in training: random split {100 * r['near_dup_share_random']:.0f}%, scaffold split {100 * r['near_dup_share_scaffold']:.0f}%")
    if len(sys.argv) > 2:
        json.dump({k: v for k, v in r.items() if k != "rows"}, open(sys.argv[2], "w"), indent=1)
