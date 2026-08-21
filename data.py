"""MNIST loading utilities (scikit-learn fetch_openml, cached locally)."""
import os

import numpy as np
from sklearn.datasets import fetch_openml

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")
N_TRAIN_SUBSET = 1000


def load_mnist_subset(n_train=N_TRAIN_SUBSET, seed=42):
    """Returns X_train, y_train_oh, X_test, y_test_oh as float64 arrays.

    Train: first `n_train` samples of the official 60k train split.
    Test:  official 10k test split.
    """
    mnist = fetch_openml("mnist_784", version=1, as_frame=False,
                         data_home=CACHE_DIR, parser="auto")
    X = mnist.data.astype(np.float64) / 255.0
    y = mnist.target.astype(np.int64)

    X_train_raw, y_train_raw = X[:60000], y[:60000]
    X_test, y_test = X[60000:], y[60000:]

    # Deterministic stratified-ish subset: take a seeded random sample.
    rng = np.random.default_rng(seed)
    idx = rng.choice(60000, size=n_train, replace=False)
    X_train, y_train = X_train_raw[idx], y_train_raw[idx]

    def onehot(labels):
        out = np.zeros((labels.size, 10))
        out[np.arange(labels.size), labels] = 1.0
        return out

    return X_train, onehot(y_train), X_test, onehot(y_test)


if __name__ == "__main__":
    Xt, yt, Xs, ys = load_mnist_subset()
    print(f"train: {Xt.shape}, test: {Xs.shape}")
    print(f"pixel range: [{Xt.min():.1f}, {Xt.max():.1f}]")
    print(f"train label counts: {np.bincount(yt.argmax(1))}")
