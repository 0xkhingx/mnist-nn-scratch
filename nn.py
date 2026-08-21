"""Two-layer neural network implemented from scratch with NumPy.

Architecture: input -> hidden (ReLU) -> output (softmax + cross-entropy).
"""
import numpy as np


# ---------------------------------------------------------------------------
# Activations and their derivatives
# ---------------------------------------------------------------------------

def sigmoid(z):
    """Numerically stable logistic sigmoid."""
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def sigmoid_grad(a):
    """Derivative of sigmoid, given the activated output a = sigmoid(z)."""
    return a * (1.0 - a)


def relu(z):
    return np.maximum(0.0, z)


def relu_grad(z):
    """Derivative of ReLU w.r.t. its pre-activation z."""
    return (z > 0.0).astype(np.float64)


def softmax(z):
    """Row-wise numerically stable softmax."""
    shifted = z - z.max(axis=1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy(probs, y_onehot):
    """Mean cross-entropy loss over the batch."""
    n = y_onehot.shape[0]
    return -np.sum(y_onehot * np.log(probs + 1e-12)) / n


# ---------------------------------------------------------------------------
# Two-layer network
# ---------------------------------------------------------------------------

class TwoLayerNN:
    def __init__(self, input_dim=784, hidden_dim=128, output_dim=10, seed=42):
        rng = np.random.default_rng(seed)
        self.W1 = rng.standard_normal((input_dim, hidden_dim)) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.standard_normal((hidden_dim, output_dim)) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(output_dim)

    # -- forward ----------------------------------------------------------

    def forward(self, X):
        """Forward propagation. Returns class probabilities and caches."""
        Z1 = X @ self.W1 + self.b1          # (N, H)
        A1 = relu(Z1)                       # (N, H)
        Z2 = A1 @ self.W2 + self.b2         # (N, C)
        probs = softmax(Z2)                 # (N, C)
        cache = {"X": X, "Z1": Z1, "A1": A1, "probs": probs}
        return probs, cache

    # -- backward ---------------------------------------------------------

    def backward(self, cache, y_onehot):
        """Backpropagation. Returns gradients for W1, b1, W2, b2."""
        X, A1, probs = cache["X"], cache["A1"], cache["probs"]
        n = y_onehot.shape[0]

        dZ2 = (probs - y_onehot) / n            # (N, C)  softmax+CE combined
        dW2 = A1.T @ dZ2                        # (H, C)
        db2 = dZ2.sum(axis=0)                   # (C,)
        dA1 = dZ2 @ self.W2.T                   # (N, H)
        dZ1 = dA1 * relu_grad(cache["Z1"])      # (N, H)
        dW1 = X.T @ dZ1                         # (D, H)
        db1 = dZ1.sum(axis=0)                   # (H,)
        return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    # -- parameters / updates ----------------------------------------------

    def params(self):
        return [self.W1, self.b1, self.W2, self.b2]

    def sgd_step(self, grads, lr):
        self.W1 -= lr * grads["dW1"]
        self.b1 -= lr * grads["db1"]
        self.W2 -= lr * grads["dW2"]
        self.b2 -= lr * grads["db2"]

    # -- training loop ------------------------------------------------------

    def train(self, X, y_onehot, epochs=150, batch_size=64, lr=0.5,
              X_test=None, y_test=None, seed=42, verbose=True):
        rng = np.random.default_rng(seed)
        n = X.shape[0]
        history = []
        for epoch in range(1, epochs + 1):
            idx = rng.permutation(n)
            for start in range(0, n, batch_size):
                mb = idx[start:start + batch_size]
                probs, cache = self.forward(X[mb])
                grads = self.backward(cache, y_onehot[mb])
                self.sgd_step(grads, lr)

            if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
                probs, _ = self.forward(X)
                loss = cross_entropy(probs, y_onehot)
                entry = {"epoch": epoch, "loss": loss,
                         "train_acc": accuracy(self.predict(X), y_onehot)}
                if X_test is not None:
                    entry["test_acc"] = accuracy(self.predict(X_test), y_test)
                history.append(entry)
                if verbose:
                    msg = (f"epoch {entry['epoch']:3d}  loss {loss:.4f}  "
                           f"train_acc {entry['train_acc']:.4f}")
                    if "test_acc" in entry:
                        msg += f"  test_acc {entry['test_acc']:.4f}"
                    print(msg)
        return history

    # -- inference ----------------------------------------------------------

    def predict(self, X):
        probs, _ = self.forward(X)
        return probs

    def predict_labels(self, X):
        return self.predict(X).argmax(axis=1)


def accuracy(pred_probs_or_labels, y_onehot):
    labels = (pred_probs_or_labels.argmax(axis=1)
              if pred_probs_or_labels.ndim == 2 else pred_probs_or_labels)
    return float((labels == y_onehot.argmax(axis=1)).mean())


# ---------------------------------------------------------------------------
# Numerical gradient check
# ---------------------------------------------------------------------------

def gradient_check(model, X, y_onehot, eps=1e-5, num_checks=40, seed=0):
    """Compare analytic gradients with central-difference numerical ones."""
    rng = np.random.default_rng(seed)
    _, cache = model.forward(X)
    analytic = model.backward(cache, y_onehot)

    names = ["dW1", "db1", "dW2", "db2"]
    tensors = dict(zip(names, model.params()))
    max_rel = 0.0
    checked = 0
    for name in names:
        arr = tensors[name]
        flat = arr.ravel()
        k = min(num_checks, flat.size)
        for fi in rng.choice(flat.size, size=k, replace=False):
            orig = flat[fi]
            flat[fi] = orig + eps
            lp = cross_entropy(model.forward(X)[0], y_onehot)
            flat[fi] = orig - eps
            lm = cross_entropy(model.forward(X)[0], y_onehot)
            flat[fi] = orig
            num = (lp - lm) / (2.0 * eps)
            ana = analytic[name].ravel()[fi]
            rel = abs(num - ana) / max(abs(num) + abs(ana), 1e-8)
            max_rel = max(max_rel, rel)
            checked += 1
    return max_rel, checked
