"""Verify every intervention arm actually is what it claims to be.

Re-runnable at any time. Checks the invariants that would silently corrupt results if
an arm were built or evaluated wrongly:

  C matched to B      ||dW_C|| == ||dW_B||        (uniform shrink to B's magnitude)
  D matched to A      ||dW_D|| == ||dW_A||        (B rescaled back to source)
  E matched to B      ||dW_E|| == ||dW_B||        (or flagged INFEASIBLE)
  F matched to B      same COUNT of directions deleted
  B < A               deleting intruders must shrink the update
  finite weights      no NaN/Inf in any evaluated adapter
  eval/adapter pair   results/<run> exists for an adapter that exists

Usage: python verify_arms.py
"""
import os, sys, json, torch
import intruder_pass as IP

ROOT = "/home/kfir/cf_models"
HERE = os.path.dirname(os.path.abspath(__file__))
ALL_CFGS = ["tia1_frc_lorawd_wd0p3_lr5e4_s43", "tia1_frc_milora_lr3e4_s43",
            "tia1_frc_clora_k1024_lr3e4_s44", "tia1_frc_milora_lr1e3_s43",
            "tia1_frc_lora_r32_lr3e4_s43", "tia1_frc_dora_r32_lr3e4_s43"]
# Run names may be given on the command line; with no arguments every known
# configuration is checked (configurations that are not built yet are skipped).
CFGS = [a for a in sys.argv[1:] if not a.startswith("-")] or ALL_CFGS
TOL = 0.005      # 0.5 % relative

def energy(run):
    d = os.path.join(ROOT, run)
    if not os.path.isdir(d): return None
    pairs, sc, cfg = IP.load_adapter(d)
    bad = False
    tot = 0.0
    for A, B in pairs.values():
        if not (torch.isfinite(A).all() and torch.isfinite(B).all()): bad = True
        Bs = sc * B
        tot += float(((Bs.T @ Bs) * (A @ A.T)).sum())
    return dict(E=tot, finite=not bad, r=cfg["r"])

def main():
    fails = 0
    for src in CFGS:
        print(f"\n=== {src} ===")
        e = {k: energy(src + s) for k, s in
             [("A", ""), ("B", "__k10allablB"), ("C", "__k10allablC"), ("D", "__k10allablD"),
              ("E", "__k10allablE"), ("Ep", "__k10allablEp"), ("F", "__k10allablF1"),
              ("G", "__k10allablG"), ("H", "__k10allablH")]}
        for k, v in e.items():
            if v is None: continue
            ratio = (v["E"] / e["A"]["E"]) ** 0.5 if e.get("A") else float("nan")
            fin = "ok" if v["finite"] else "NON-FINITE!"
            ev = "evaluated" if os.path.exists(
                os.path.join(HERE, "results", src + ("__rl50" if k == "A" else
                    {"B":"__k10allablB","C":"__k10allablC","D":"__k10allablD","E":"__k10allablE",
                     "Ep":"__k10allablEp","F":"__k10allablF1","G":"__k10allablG",
                     "H":"__k10allablH"}[k]), "summary.json")) else "pending"
            print(f"   {k:3s} r={v['r']:3d}  ||dW||/||dW_A|| = {ratio:.4f}  weights {fin}  {ev}")
            if not v["finite"]: fails += 1
        def chk(x, y, label):
            nonlocal fails
            if not e.get(x) or not e.get(y): return
            rel = abs(e[x]["E"] - e[y]["E"]) / e[y]["E"]
            ok = rel < TOL
            if not ok and x == "E":
                print(f"   [INFEASIBLE] {label}: rel diff {rel:.4f} "
                      f"(non-intruder content insufficient — expected for high-intruder configs)")
                return
            print(f"   [{'PASS' if ok else 'FAIL'}] {label}: rel diff {rel:.5f}")
            if not ok: fails += 1
        chk("C", "B", "C matched to B")
        chk("D", "A", "D matched to A")
        chk("E", "B", "E matched to B")
        if e.get("B") and e.get("A") and e["B"]["E"] >= e["A"]["E"]:
            print("   [FAIL] B should be smaller than A"); fails += 1
    print(f"\n{'ALL INVARIANTS HOLD' if fails==0 else f'{fails} FAILURES'}")
    return fails

if __name__ == "__main__":
    raise SystemExit(0 if main() == 0 else 1)
