"""Baselines and candidate models. Each: fit(X, Y) then predict(X) -> Y_hat."""
import numpy as np
from .world import tanimoto


class Mean:
    name = "training mean"
    def fit(self, X, Y, **_): self.mu = Y.mean(0); return self
    def predict(self, X): return np.tile(self.mu, (len(X), 1))


class NearestAnalog:
    name = "nearest analog (Tanimoto kNN)"
    def __init__(self, k=3): self.k = k
    def fit(self, X, Y, **_): self.X, self.Y = X, Y; return self
    def predict(self, X):
        S = tanimoto(X, self.X)
        idx = np.argsort(-S, axis=1)[:, :self.k]
        w = np.take_along_axis(S, idx, 1) + 1e-6
        return (self.Y[idx] * w[..., None]).sum(1) / w.sum(1, keepdims=True)


class Ridge:
    name = "ridge on fingerprint"
    def __init__(self, lam=30.0): self.lam = lam
    def fit(self, X, Y, **_):
        self.mu = Y.mean(0)
        self.X = X
        self.a = np.linalg.solve(X @ X.T + self.lam * np.eye(len(X)), Y - self.mu)
        return self
    def predict(self, X): return self.mu + X @ self.X.T @ self.a


class Candidate:
    """Stand-in for a foundation model: blends what it really learned about mechanisms
    (pharmacophore -> mechanism map with error 'skill') with analog look-up ('memo')."""
    name = "candidate model"
    def __init__(self, bio, rng, skill, memo):
        self.bio, self.memo = bio, memo
        self.A_hat = bio.A + rng.normal(0, skill, bio.A.shape)
        self.knn = NearestAnalog(1)
    def fit(self, X, Y, **_):
        self.mu = Y.mean(0); self.knn.fit(X, Y); return self
    def predict(self, X):
        mech = self.bio.base + (X[:, self.bio.ph_bits] @ self.A_hat) @ self.bio.W
        return (1 - self.memo) * mech + self.memo * self.knn.predict(X)


BASELINES = [Mean, NearestAnalog, Ridge]
