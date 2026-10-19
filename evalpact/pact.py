"""The evaluation contract: frozen before anyone sees results, then applied mechanically."""
import hashlib, json
import numpy as np
from .world import tanimoto

CONTRACT = {
    "endpoint": "per-compound Pearson of predicted vs measured response change (delta from training mean) over all genes",
    "split": "scaffold-grouped: 20% of scaffold families held out; no family on both sides",
    "baselines": ["training mean", "nearest analog (Tanimoto kNN)", "ridge on fingerprint"],
    "compare_to": "the best baseline on the held-out set",
    "decision": "BEATS only if the one-sided 95% lower bound of (model - best baseline), bootstrapped over held-out families, is > 0",
    "leakage_checks": {"tanimoto_near_duplicate": 0.8, "max_share_near_duplicates": 0.05, "shared_families": 0},
}


def contract_hash(c=CONTRACT):
    return hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()[:16]


def delta_pearson(P, Y, mu):
    a, b = P - mu, Y - mu
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    den = np.sqrt((a * a).sum(1) * (b * b).sum(1))
    return np.where(den > 1e-12, (a * b).sum(1) / np.where(den > 1e-12, den, 1), 0.0)


def split(rng, fam, kind, frac=0.2):
    if kind == "random":
        test = rng.random(len(fam)) < frac
    else:
        fams = np.unique(fam)
        held = rng.choice(fams, max(1, int(round(frac * len(fams)))), replace=False)
        test = np.isin(fam, held)
    return ~test, test


def leakage(X, fam, plates, tr, te, thr=0.8):
    S = tanimoto(X[te], X[tr]).max(1)
    return dict(near_duplicate_share=float((S >= thr).mean()), median_max_tanimoto=float(np.median(S)),
                shared_families=int(len(set(fam[te]) & set(fam[tr]))),
                shared_plates=int(len(set(plates[te]) & set(plates[tr]))))


def scores(models, X, Y, tr, te):
    mu = Y[tr].mean(0)
    out = {}
    for name, m in models.items():
        m.fit(X[tr], Y[tr])
        out[name] = delta_pearson(m.predict(X[te]), Y[te], mu)
    return out


def verdict(sc, fam_te, cand="candidate model", rng=None, reps=2000, alpha=0.05):
    base = {k: v.mean() for k, v in sc.items() if k != cand}
    best = max(base, key=base.get)
    d = sc[cand] - sc[best]
    fams = np.unique(fam_te)
    rng = rng or np.random.default_rng(0)
    idx = {f: np.where(fam_te == f)[0] for f in fams}
    boots = []
    for _ in range(reps):
        pick = rng.choice(fams, fams.size)
        boots.append(np.concatenate([d[idx[f]] for f in pick]).mean())
    lb = float(np.quantile(boots, alpha))
    return dict(model=float(sc[cand].mean()), best_baseline=best, baseline=float(base[best]), diff=float(d.mean()),
                lower_bound=lb, beats=lb > 0)
