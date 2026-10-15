"""sign_criterion 审计（2026-10-07）：
  C1 证书不等式独立重推（P 用 D 下界、N 必须用 D 上界 D̂+ν）
     ——与已落盘证书对比，检测实现/陈述错误
  C2 ε̄ 闭式并集上界 vs 蒙特卡洛直接测量（P(u 在 A 下有边|腐蚀)）
  C3 D̂<0 声明复验 + ν_S 重算（从 JSON rows）
  C4 (S)-版证书健全性（修正 N 条件后）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")
import sign_predict as sp

R = Path("results")
ok_all = True


def check(name, cond, detail=""):
    global ok_all
    if not cond:
        ok_all = False
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")


print("=" * 78)
for tag in ("cora", "dblp", "cora_appnp"):
    print(f"\n### {tag}")
    sc = json.loads((R / f"signcriterion_{tag}.json").read_text(
        encoding="utf-8"))
    rows = sc["rows"]
    nuS = {tuple(map(float, k.split("|"))): v
           for k, v in sc["nu_S"].items()}
    dBphi = {float(k): v for k, v in sc["delta_B_by_phi"].items()}

    # C1: 重推证书条件
    n_P_impl_wrongN = 0
    cert_re = {"S": [], "strict": []}
    for r in rows:
        p, phi, T = r["p"], r["phi"], r["T"]
        # 与实现一致：隔离松弛 = sqrt(ε̄·(ed2+3se))（Cauchy–Schwarz）
        ed2, se2 = sc["a_ed2"][f"{p}"][str(T)]
        slack_iso = float(np.sqrt(sc["eps_bar"][f"{p}"]
                                  * (ed2 + 3 * se2)))
        dB = dBphi[phi]
        G_lb = (1 - 2 * phi) - slack_iso - (r["beta_mf"] + dB)
        G_ub = (1 - 2 * phi) + slack_iso - (r["beta_mf"] - dB)
        for kind, dslack in (("S", nuS[(p, T)]),
                             ("strict", sc["d_cln_strict"])):
            D_lb = r["d_mf"] - dslack
            D_ub = r["d_mf"] + dslack
            P_ok = p * G_lb + (1 - p) * D_lb > 0        # Δ ≥ pG_lb+(1−p)D_lb
            N_ok = p * G_ub + (1 - p) * D_ub < 0        # Δ ≤ pG_ub+(1−p)D_ub
            cert_re[kind].append(dict(P=bool(P_ok), N=bool(N_ok),
                                      meas=r["meas"]))
    # 与已实现证书对比（注意：已实现 N 用了 D_lb）
    impl = json.loads((R / f"signcriterion_{tag}.json").read_text(
        encoding="utf-8"))["certificates"]
    for kind in ("S", "strict"):
        n_impl = sum(1 for c in impl if c["kind"] == kind and c["N"])
        n_re = sum(1 for c in cert_re[kind] if c["N"])
        diff = n_impl - n_re
        check(f"C1 {kind} 版 N 证书数（实现 vs 正确推导）",
              diff == 0, f"(实现 {n_impl}, 正确 {n_re}, "
                         f"实现多出 {diff} 个 → N 条件有 bug)" if diff
              else "(一致)")
        nP_impl = sum(1 for c in impl if c["kind"] == kind and c["P"])
        nP_re = sum(1 for c in cert_re[kind] if c["P"])
        check(f"C1 {kind} 版 P 证书数", nP_impl == nP_re,
              f"(实现 {nP_impl}, 推导 {nP_re})")

    # C4: 修正后的健全性
    for kind in ("S", "strict"):
        sub = cert_re[kind]
        falseP = sum(1 for c in sub if c["P"] and c["meas"] <= 0)
        falseN = sum(1 for c in sub if c["N"] and c["meas"] >= 0)
        unamb = sum(1 for c in sub if abs(c["meas"]) > 0.001)
        cov = sum(1 for c in sub if (c["P"] or c["N"])
                  and abs(c["meas"]) > 0.001)
        check(f"C4 修正N后 [{kind}] 无假证书", falseP == 0 and falseN == 0,
              f"(假P {falseP}, 假N {falseN}; 覆盖 {cov}/{unamb})")

    # C3: D̂ 与实测 D 同号（算子无关；热流 D̂<0、APPNP 高 φ 为正）
    dsign = all((r["d_mf"] < 0) == (r["D_meas"] < 0)
                or abs(r["d_mf"]) < 0.002 or abs(r["D_meas"]) < 0.002
                for r in rows)
    nus = max(abs(r["D_meas"] - r["d_mf"]) for r in rows)
    check("C3 sign(D̂)=sign(D_meas)（近零除外）", dsign)
    check("C3 ν_S 复算一致",
          abs(nus - max(nuS.values())) < 1e-9,
          f"(max|D−D̂| = {nus:.4f})")

print("\n### C2 ε̄ 闭式 vs 蒙特卡洛（cora, p=0.1/0.2, 60 实现）")
import sign_criterion as scr
ds = sp.Dataset("cora")
for p in (0.1, 0.2):
    closed = scr.eps_bar(ds, p)
    w0 = ds.expected_da(p)
    q_rate, sgn = np.abs(w0), np.sign(w0)
    rng = np.random.default_rng(31337)
    hit = 0
    ncor = 0
    for _ in range(60):
        pred = sp.corrupt_labels(ds.labels, ds.C, p, rng)
        corrupted = pred != ds.labels
        wA = ds.all_hom(pred)
        # u 有幸存边 ⟺ 某条 wA=1 的边包含 u
        alive = np.zeros(ds.n, dtype=bool)
        alive[ds.rows[wA[ds.owner] > 0]] = True
        hit += int((alive & corrupted).sum())
        ncor += int(corrupted.sum())
    mc = hit / ncor
    check(f"C2 ε̄(p={p}) 闭式 ≥ MC",
          closed >= mc - 0.005,
          f"(闭式 {closed:.4f} vs MC {mc:.4f})")

print("\n" + "=" * 78)
print("审计总结:", "全部通过" if ok_all else "存在 FAIL（N 条件 bug 需修复）")
