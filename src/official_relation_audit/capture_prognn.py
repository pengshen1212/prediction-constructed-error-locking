"""Observer wrapper around the author Pro-GNN / DeepRobust implementation."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

import numpy as np
import torch
import scipy.linalg


OUT_DIR = Path(os.environ["PAPER2_AUDIT_OUT"]).resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if hasattr(value, "toarray"):
        return value.toarray()
    return np.asarray(value)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: capture_prognn.py TRAIN.PY [official train.py arguments]")
    train_path = Path(sys.argv[1]).resolve()
    official_root = train_path.parent
    deep_robust_root = Path(os.environ["PAPER2_DEEPROBUST_ROOT"]).resolve()
    sys.path.insert(0, str(deep_robust_root))

    # DeepRobust's pinned Gensim dependency still imports this removed SciPy alias.
    if not hasattr(scipy.linalg, "triu"):
        scipy.linalg.triu = np.triu

    from deeprobust.graph.defense import ProGNN

    original_test = ProGNN.test
    original_fit = ProGNN.fit

    def observed_fit(self, features, adj, labels, idx_train, idx_val, *args, **kwargs):
        self.audit_idx_train = np.asarray(idx_train, dtype=np.int64)
        self.audit_idx_val = np.asarray(idx_val, dtype=np.int64)
        return original_fit(self, features, adj, labels, idx_train, idx_val, *args, **kwargs)

    ProGNN.fit = observed_fit

    def observed_test(self, features, labels, idx_test):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(features, self.best_graph)
        seed = int(self.args.seed)
        path = OUT_DIR / f"prognn_seed-{seed}.npz"
        np.savez_compressed(
            path,
            relation=self.best_graph.detach().cpu().numpy(),
            raw_relation=self.estimator.estimated_adj.detach().cpu().numpy(),
            logits=logits.detach().cpu().numpy(),
            labels=labels.detach().cpu().numpy(),
            idx_train=self.audit_idx_train,
            idx_val=self.audit_idx_val,
            idx_test=idx_test.detach().cpu().numpy() if isinstance(idx_test, torch.Tensor) else np.asarray(idx_test),
            seed=np.asarray(seed),
        )
        print(f"[audit observer] saved official Pro-GNN relation/logits: {path}")
        return original_test(self, features, labels, idx_test)

    ProGNN.test = observed_test
    official_args = sys.argv[2:]
    sys.argv = [str(train_path), *official_args]
    previous_cwd = Path.cwd()
    os.chdir(official_root)
    try:
        namespace = runpy.run_path(str(train_path), run_name="__main__")
    finally:
        os.chdir(previous_cwd)

    if "perturbed_adj" in namespace:
        seed = int(namespace["args"].seed)
        input_dir = OUT_DIR.parent / "input_graph"
        input_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            input_dir / f"prognn_cora_seed-{seed}.npz",
            original_relation=_to_numpy(namespace["adj"]),
            input_relation=_to_numpy(namespace["perturbed_adj"]),
            labels=_to_numpy(namespace["labels"]),
            idx_train=np.asarray(namespace["idx_train"]),
            idx_val=np.asarray(namespace["idx_val"]),
            idx_test=np.asarray(namespace["idx_test"]),
        )


if __name__ == "__main__":
    main()
