"""Train the from-scratch NumPy NN on a 1000-sample MNIST subset."""
import json
import os
import time

from data import load_mnist_subset
from nn import TwoLayerNN, accuracy, gradient_check


def main():
    print("Loading MNIST (cached after first run)...")
    X_train, y_train, X_test, y_test = load_mnist_subset()
    print(f"train subset: {X_train.shape[0]} samples, "
          f"test set: {X_test.shape[0]} samples\n")

    # --- sanity check: gradients vs numerical approximation -------------
    model = TwoLayerNN(input_dim=784, hidden_dim=8, output_dim=10)
    max_rel, checked = gradient_check(
        model, X_train[:5], y_train[:5], eps=1e-5, num_checks=40)
    print(f"Gradient check: max rel error {max_rel:.2e} over {checked} params "
          f"({'PASS' if max_rel < 1e-6 else 'FAIL'})\n")

    # --- full training ----------------------------------------------------
    model = TwoLayerNN(input_dim=784, hidden_dim=128, output_dim=10, seed=42)
    t0 = time.time()
    model.train(X_train, y_train, epochs=150, batch_size=64, lr=0.5,
                X_test=X_test, y_test=y_test)
    elapsed = time.time() - t0

    test_acc = accuracy(model.predict(X_test), y_test)
    train_acc = accuracy(model.predict(X_train), y_train)
    print(f"\nFinal: train_acc {train_acc:.4f}  test_acc {test_acc:.4f}  "
          f"({elapsed:.1f}s)")

    results = {"framework": "numpy", "test_acc": round(test_acc, 4),
               "train_acc": round(train_acc, 4), "seconds": round(elapsed, 1)}
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results_scratch.json")
    with open(out, "w") as f:
        json.dump(results, f)

    if test_acc > 0.80:
        print(f"DELIVERABLE MET: test accuracy {test_acc*100:.2f}% > 80%")
    else:
        print(f"WARNING: test accuracy {test_acc*100:.2f}% <= 80%")


if __name__ == "__main__":
    main()
