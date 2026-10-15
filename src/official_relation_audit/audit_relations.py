"""Compute frozen-relation diagnostics from observer NPZ files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp


def load_run(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def as_indices(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.int64).reshape(-1)


def audit_one(path: Path, steps: list[int], nodes_file) -> dict:
    run = load_run(path)
    W = np.asarray(run["relation"], dtype=np.float64)
    logits = np.asarray(run["logits"], dtype=np.float64)
    labels = np.asarray(run["labels"], dtype=np.int64).reshape(-1)
    test = as_indices(run["idx_test"])
    if W.ndim != 2 or W.shape[0] != W.shape[1] or logits.shape[0] != W.shape[0]:
        raise ValueError(f"incompatible relation/logit dimensions in {path}")
    if labels.shape[0] != W.shape[0]:
        raise ValueError(f"labels do not cover all nodes in {path}")
    if not np.isfinite(W).all() or not np.isfinite(logits).all():
        raise ValueError(f"nonfinite official output in {path}")
    if W.min(initial=0.0) < -1e-7:
        raise ValueError(f"negative weights in propagation relation: {W.min()} ({path})")
    W[W < 0] = 0
    row_sums = W.sum(axis=1)
    if np.any(row_sums <= 1e-12):
        raise ValueError(f"zero-outdegree rows; refusing to invent self loops ({path})")

    K = sp.csr_matrix(W / row_sums[:, None])
    source = logits.argmax(axis=1)
    source_correct = source == labels
    nclass = logits.shape[1]
    onehot = sp.csr_matrix((np.ones(len(source)), (np.arange(len(source)), source)),
                           shape=(len(source), nclass))
    current = onehot
    metrics = []

    for t in range(1, max(steps) + 1):
        current = K @ current
        if t not in steps:
            continue
        scores = current.toarray()
        propagated = scores.argmax(axis=1)
        same_class_mass = scores[np.arange(len(source)), source]
        changed = propagated != source
        corrected = (~source_correct) & (propagated == labels)
        correct_to_wrong = source_correct & (propagated != labels)
        wrong_to_wrong = (~source_correct) & changed & (propagated != labels)
        certified = same_class_mass > 0.5 + 1e-12
        ids = test
        n_test = len(ids)
        n_wrong = int((~source_correct[ids]).sum())
        n_right = int(source_correct[ids].sum())
        entry = {
            "step": t,
            "n_test": n_test,
            "model_test_accuracy": float(source_correct[ids].mean()),
            "propagated_test_accuracy": float((propagated[ids] == labels[ids]).mean()),
            "lock_certificate_coverage": float(certified[ids].mean()),
            "mean_source_class_mass": float(same_class_mass[ids].mean()),
            "initial_errors": n_wrong,
            "corrections": int(corrected[ids].sum()),
            "correction_rate_among_initial_errors": float(corrected[ids].sum() / n_wrong) if n_wrong else None,
            "correct_to_wrong": int(correct_to_wrong[ids].sum()),
            "harmful_change_rate_among_initially_correct": float(correct_to_wrong[ids].sum() / n_right) if n_right else None,
            "wrong_to_wrong": int(wrong_to_wrong[ids].sum()),
            "all_class_changes": int(changed[ids].sum()),
            "class_change_rate": float(changed[ids].mean()),
            "certified_nodes_that_changed": int((certified[ids] & changed[ids]).sum()),
        }
        metrics.append(entry)

        for u in ids:
            nodes_file.writerow({
                "method": path.stem.split("_seed-")[0],
                "seed": int(np.asarray(run["seed"]).item()),
                "node": int(u),
                "step": t,
                "true_class": int(labels[u]),
                "source_class": int(source[u]),
                "propagated_class": int(propagated[u]),
                "source_correct": bool(source_correct[u]),
                "same_class_mass": float(same_class_mass[u]),
                "certified": bool(certified[u]),
                "changed": bool(changed[u]),
                "corrected": bool(corrected[u]),
                "correct_to_wrong": bool(correct_to_wrong[u]),
                "wrong_to_wrong": bool(wrong_to_wrong[u]),
            })

    return {
        "method": path.stem.split("_seed-")[0],
        "seed": int(np.asarray(run["seed"]).item()),
        "relation_shape": list(W.shape),
        "relation_nnz": int(np.count_nonzero(W)),
        "relation_density": float(np.count_nonzero(W) / W.size),
        "relation_min": float(W.min()),
        "relation_max": float(W.max()),
        "test_indices": test.tolist(),
        "steps": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 2, 5, 10])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    files = sorted(args.input.glob("*_seed-*.npz"))
    if not files:
        raise SystemExit(f"no captured model runs under {args.input}")

    records = []
    nodes_path = args.output / "per_node.csv"
    with nodes_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["method", "seed", "node", "step", "true_class", "source_class",
                  "propagated_class", "source_correct", "same_class_mass", "certified",
                  "changed", "corrected", "correct_to_wrong", "wrong_to_wrong"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for path in files:
            records.append(audit_one(path, args.steps, writer))

    aggregates = {}
    for method in sorted({r["method"] for r in records}):
        by_step = {}
        for step in args.steps:
            vals = [next(x for x in r["steps"] if x["step"] == step)
                    for r in records if r["method"] == method]
            by_step[str(step)] = {
                "n_runs": len(vals),
                "metrics": {
                    key: {
                        "mean": float(np.mean([v[key] for v in vals if v[key] is not None])),
                        "sd_across_seeds": float(np.std([v[key] for v in vals if v[key] is not None], ddof=1)) if len([v for v in vals if v[key] is not None]) > 1 else None,
                        "per_seed": [v[key] for v in vals],
                    }
                    for key in vals[0] if key not in ("step", "n_test")
                },
            }
        aggregates[method] = by_step

    with (args.output / "audit.json").open("w", encoding="utf-8") as handle:
        json.dump({"runs": records, "aggregate": aggregates}, handle, indent=2)
        handle.write("\n")
    print(f"Audited {len(records)} runs; wrote {args.output / 'audit.json'} and {nodes_path}")


if __name__ == "__main__":
    main()
