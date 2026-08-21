"""PyTorch reference implementation for comparison.

Identical architecture and hyperparameters to the NumPy version:
784 -> 128 (ReLU) -> 10, softmax cross-entropy, SGD, batch 64, lr 0.5, 150 epochs.
"""
import json
import os
import time

import torch
import torch.nn as nn

from data import load_mnist_subset


class TwoLayerTorch(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=128, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def main():
    torch.manual_seed(42)
    print("Loading MNIST...")
    X_train, y_train_oh, X_test, y_test_oh = load_mnist_subset()

    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train_oh.argmax(1)).long()
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test_oh.argmax(1)).long()

    model = TwoLayerTorch()
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=0.5)

    n = X_train.shape[0]
    batch_size = 64
    epochs = 150
    gen = torch.Generator().manual_seed(42)

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=gen)
        for start in range(0, n, batch_size):
            mb = perm[start:start + batch_size]
            opt.zero_grad()
            loss = criterion(model(X_train[mb]), y_train[mb])
            loss.backward()
            opt.step()

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                tr_acc = (model(X_train).argmax(1) == y_train).float().mean().item()
                te_acc = (model(X_test).argmax(1) == y_test).float().mean().item()
            print(f"epoch {epoch:3d}  loss {loss.item():.4f}  "
                  f"train_acc {tr_acc:.4f}  test_acc {te_acc:.4f}")
    elapsed = time.time() - t0

    model.eval()
    with torch.no_grad():
        train_acc = (model(X_train).argmax(1) == y_train).float().mean().item()
        test_acc = (model(X_test).argmax(1) == y_test).float().mean().item()
    print(f"\nFinal: train_acc {train_acc:.4f}  test_acc {test_acc:.4f}  "
          f"({elapsed:.1f}s)")

    # --- side-by-side comparison -----------------------------------------
    results = {"framework": "pytorch", "test_acc": round(test_acc, 4),
               "train_acc": round(train_acc, 4), "seconds": round(elapsed, 1)}
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "results_torch.json")
    with open(out, "w") as f:
        json.dump(results, f)

    scratch_path = os.path.join(here, "results_scratch.json")
    print("\n=== Comparison: from-scratch NumPy vs PyTorch ===")
    print(f"{'':12s}{'test_acc':>10s}{'train_acc':>11s}{'time(s)':>9s}")
    if os.path.exists(scratch_path):
        s = json.load(open(scratch_path))
        print(f"{'numpy':12s}{s['test_acc']:>10.4f}{s['train_acc']:>11.4f}"
              f"{s['seconds']:>9.1f}")
    else:
        print("numpy      (run train_scratch.py first)")
    print(f"{'pytorch':12s}{test_acc:>10.4f}{train_acc:>11.4f}{elapsed:>9.1f}")


if __name__ == "__main__":
    main()
