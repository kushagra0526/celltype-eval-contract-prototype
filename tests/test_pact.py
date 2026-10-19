import os, sys, unittest
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evalpact import world, pact, models


class T(unittest.TestCase):
    def test_tanimoto(self):
        a = np.array([[1, 1, 0, 0.]]); b = np.array([[1, 0, 1, 0.], [1, 1, 0, 0.]])
        self.assertTrue(np.allclose(pact.tanimoto(a, b) if hasattr(pact, "tanimoto") else world.tanimoto(a, b), [[1 / 3, 1.0]]))

    def test_delta_pearson(self):
        mu = np.zeros(5); y = np.array([[1, 2, 3, 4, 5.]])
        self.assertAlmostEqual(pact.delta_pearson(2 * y, y, mu)[0], 1.0)
        self.assertAlmostEqual(pact.delta_pearson(np.zeros((1, 5)), y, mu)[0], 0.0)

    def test_scaffold_split_has_no_shared_family(self):
        rng = np.random.default_rng(0)
        fam = np.repeat(np.arange(30), 4)
        tr, te = pact.split(rng, fam, "scaffold")
        self.assertFalse(set(fam[tr]) & set(fam[te]))

    def test_leakage_sees_analogs(self):
        rng = np.random.default_rng(1)
        bio = world.Biology(rng, world.DEV)
        X, fam, M = bio.compounds(rng, 40)
        _, plates = bio.measure(rng, M)
        tr, te = pact.split(rng, fam, "random")
        tr2, te2 = pact.split(rng, fam, "scaffold")
        a, b = pact.leakage(X, fam, plates, tr, te), pact.leakage(X, fam, plates, tr2, te2)
        self.assertGreater(a["median_max_tanimoto"], b["median_max_tanimoto"] + 0.3)
        self.assertEqual(b["shared_families"], 0)

    def test_verdict_bootstrap(self):
        fam = np.repeat(np.arange(20), 5)
        sc = {"candidate model": np.full(100, 0.6) + np.random.default_rng(2).normal(0, 0.05, 100), "ridge": np.full(100, 0.4)}
        self.assertTrue(pact.verdict(sc, fam)["beats"])
        sc["candidate model"] = np.full(100, 0.4) + np.random.default_rng(3).normal(0, 0.2, 100)
        self.assertFalse(pact.verdict(sc, fam)["beats"])

    def test_contract_hash_stable(self):
        self.assertEqual(pact.contract_hash(), pact.contract_hash(dict(pact.CONTRACT)))


if __name__ == "__main__":
    unittest.main()
