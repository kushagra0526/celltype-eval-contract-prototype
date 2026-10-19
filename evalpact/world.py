"""A simulated compound screen with known truth.

Compounds come in scaffold families (analog series) and are 1024-bit fingerprints: each
family has core bits its members mostly share. A compound's mechanism vector comes partly
from pharmacophore bits shared ACROSS families (learnable, generalizes to new scaffolds) and
partly from its family (only learnable by having seen analogs). Response = mechanism @ gene
loadings + plate effect + noise, over 200 genes.
"""
import numpy as np

BITS, GENES, K, NPH = 1024, 200, 8, 40
DEV = dict(fam=(60, 80), fam_size=(3, 14), core=40, keep=0.93, extra=5, ph_per_fam=4, fam_effect=1.0,
           ph_effect=1.0, noise=0.6, plate=0.3, cmpd_dev=0.35)


class Biology:
    def __init__(self, rng, cfg):
        self.cfg = cfg
        self.A = rng.normal(0, 1, (NPH, K)) * (rng.random((NPH, K)) < 0.25)   # pharmacophore -> mechanism
        self.W = rng.normal(0, 1, (K, GENES)) * (rng.random((K, GENES)) < 0.3)  # mechanism -> genes
        self.base = rng.normal(0, 0.5, GENES)                                  # response shared by most compounds
        self.ph_bits = rng.choice(BITS, NPH, replace=False)

    def compounds(self, rng, n_fam, first_fam=0):
        c = self.cfg
        X, fam, M = [], [], []
        for f in range(n_fam):
            core = rng.choice(BITS, c["core"], replace=False)
            ph = rng.choice(NPH, c["ph_per_fam"], replace=False)
            fam_m = rng.normal(0, c["fam_effect"], K) * (rng.random(K) < 0.3)
            for _ in range(rng.integers(*c["fam_size"])):
                x = np.zeros(BITS)
                x[core[rng.random(core.size) < c["keep"]]] = 1
                x[rng.choice(BITS, c["extra"], replace=False)] = 1
                on = ph[rng.random(ph.size) < 0.8]
                x[self.ph_bits[on]] = 1
                m = c["ph_effect"] * x[self.ph_bits] @ self.A + fam_m + rng.normal(0, c["cmpd_dev"], K) * (rng.random(K) < 0.3)
                X.append(x); fam.append(first_fam + f); M.append(m)
        return np.array(X), np.array(fam), np.array(M)

    def measure(self, rng, M, n_plates=12):
        c = self.cfg
        plates = rng.integers(0, n_plates, len(M))
        pe = rng.normal(0, c["plate"], (n_plates, GENES))
        Y = self.base + M @ self.W + pe[plates] + rng.normal(0, c["noise"], (len(M), GENES))
        return Y, plates


def tanimoto(A, B):
    inter = A @ B.T
    return inter / (A.sum(1)[:, None] + B.sum(1)[None, :] - inter + 1e-9)
