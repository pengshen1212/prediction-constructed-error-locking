"""Reanalyze saved propagation trajectories; no fitted rules or new simulation."""
from pathlib import Path
import ast
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "experiments/prob2-relation-error/results"

def summarize(cell, n_boot=2000, seed=99):
    a = {k: np.asarray(v, dtype=float) for k, v in cell.items()}
    n = len(a["mA"])
    assert n > 1 and all(v.shape == (n,) for v in a.values())
    assert all(np.isfinite(v).all() for v in a.values())
    phat = a["phat"]
    assert ((phat > 0) & (phat < 1)).all()
    cor = phat * (a["mA_cor"] - a["mB_cor"])
    clean = (1-phat) * (a["mA_cln"] - a["mB_cln"])
    margin = a["mA"] - a["mB"]
    accuracy = a["aA"] - a["aB"]
    residual = float(np.max(np.abs(margin-cor-clean)))
    assert residual < 1e-10, f"Invalid node-subset decomposition: {residual}"
    for model in ("A", "B"):
        reconstructed = phat*a[f"m{model}_cor"]+(1-phat)*a[f"m{model}_cln"]
        assert np.max(np.abs(a[f"m{model}"]-reconstructed)) < 1e-10
    idx = np.random.default_rng(seed).integers(0, n, (n_boot, n))
    def ci(values):
        return np.quantile(values[idx].mean(axis=1), [0.025, 0.975]).tolist()
    return dict(n_realizations=n, realized_corruption=float(phat.mean()),
                margin_difference=float(margin.mean()), margin_ci=ci(margin),
                accuracy_difference=float(accuracy.mean()), accuracy_ci=ci(accuracy),
                corrupted_contribution=float(cor.mean()), corrupted_ci=ci(cor),
                clean_contribution=float(clean.mean()), clean_ci=ci(clean),
                max_decomposition_residual=residual,
                conditional_means={k:float(a[k].mean()) for k in
                    ("mA_cor", "mB_cor", "mA_cln", "mB_cln")})

def main():
    out = dict(protocol="paired saved-trajectory reanalysis, 2026-10-08",
               n_boot=2000, bootstrap_seed=99,
               weighting="realized phat in each realization, then average",
               scope="fixed datasets; no training, propagation rerun, or fitted prediction rule",
               sources={}, cells=[], endpoint_contrasts=[])
    for tag in ("cora", "cora_appnp", "dblp_appnp"):
        path = RESULTS / f"signpred_{tag}_eval_raw.json"
        out["sources"][tag] = dict(path=str(path.relative_to(ROOT)),
                                  sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        raw = json.loads(path.read_text(encoding="utf-8"))
        for p in (0.1, 0.2):
            truth = raw[str(("sweep", p, "couple", 0.0, None))]["20"]
            coupled = raw[str(("sweep", p, "couple", 1.0, None))]["20"]
            np.testing.assert_array_equal(truth["phat"], coupled["phat"])
            dt = np.asarray(truth["mA"])-np.asarray(truth["mB"])
            dc = np.asarray(coupled["mA"])-np.asarray(coupled["mB"])
            contrast = dc-dt
            idx = np.random.default_rng(99).integers(0, len(dt), (2000, len(dt)))
            out["endpoint_contrasts"].append(dict(
                tag=tag, p=p, depth=20, coupled_minus_truth=float(contrast.mean()),
                ci=np.quantile(contrast[idx].mean(axis=1), [0.025,0.975]).tolist()))
        for key, record in raw.items():
            kind, p, mode, phi, q = ast.literal_eval(key)
            selected = ((kind=="sweep" and mode=="couple" and phi in (0., 1.)
                         and p in (0.1, 0.2))
                        or (tag=="cora" and kind=="indep" and mode=="indep"
                            and p==0.1 and q==0.1))
            if not selected:
                continue
            init = ("truth" if phi==0 else "coupled") if mode=="couple" else "independent"
            depths = (2,5,10,20) if tag=="cora" and p==0.1 else (20,)
            for t in depths:
                out["cells"].append(dict(tag=tag, p=p, initialization=init, q=q,
                                         depth=t, source_cell=key,
                                         **summarize(record[str(t)])))
    destination = ROOT / "mechanism-reanalysis.json"
    destination.write_text(json.dumps(out, indent=2)+"\n", encoding="utf-8")
    print(f"Verified and summarized {len(out['cells'])} saved cells: {destination}")
    print(f"Maximum decomposition residual: {max(c['max_decomposition_residual'] for c in out['cells']):.3g}")
    for c in out["cells"]:
        if c["depth"]==20 and c["p"]==0.1:
            print(c["tag"], c["initialization"],
                  "margin", round(c["margin_difference"],6), c["margin_ci"],
                  "accuracy", round(c["accuracy_difference"],6), c["accuracy_ci"],
                  "components", round(c["corrupted_contribution"],6),
                  round(c["clean_contribution"],6))
if __name__ == "__main__":
    main()
