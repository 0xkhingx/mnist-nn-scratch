# mnist-nn-scratch — 2-layer neural net from scratch in NumPy

A deliberately small neural network that proves the fundamentals still matter when the framework layer is removed: forward pass, backpropagation, and SGD written by hand in NumPy, then benchmarked against a PyTorch baseline on the same data split.

**Result: ~89.7% MNIST test accuracy, near PyTorch parity** (see `results_scratch.json` vs `results_torch.json`).

## Files

| File | What it is |
|---|---|
| `nn.py` | `TwoLayerNN` (784 → 128 → 10, seed 42), `accuracy`, numerical `gradient_check` |
| `data.py` | 1000-sample MNIST subset via sklearn `fetch_openml` (cached after first run) |
| `train_scratch.py` | Gradient sanity check (must pass < 1e-6) → 150 epochs, batch 64, lr 0.5 → writes `results_scratch.json` |
| `train_torch.py` | Same setup in PyTorch → writes `results_torch.json` for the fair comparison |

## Run it

```bash
pip install -r requirements.txt
python train_scratch.py   # numpy net, ~80%+ gate printed at the end
python train_torch.py     # pytorch baseline
```

## Why it exists

To understand training mechanics well enough to reproduce them by hand — then compare against the baseline without hand-waving the math. Shape bugs along the way turned into a much sharper intuition for how training actually fails.

## License

MIT — see [LICENSE](LICENSE).
