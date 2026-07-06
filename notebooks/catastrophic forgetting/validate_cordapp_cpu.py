"""
CPU-only validation for cordapp_init.py (CorDA++). No model / no GPU — small synthetic matrices only.

Checks (prints PASS/FAIL + numbers per check; exits 0 iff ALL pass):
  1. compactness(C) matches the closed form pi=sqrt(d_out*sigma_max)/sigma_min on known SPD matrices
     (diagonal + a random rotation of the same spectrum), and honors the svals= operand override.
  2. kpm_BAR reuses the static CorDA decomposition faithfully: kpm_BAR(W,C,r) == corda_init.corda_kpa_BAR.
  3. KPM init is loss-preserving at a per-layer rank AND at scaling=2 (alpha=2r) — the faithful setting:
     W_res + scaling*(B/scaling)@A == W0 within fp32 tol.
  4. select_covariances picks, per layer, the most-compact candidate (argmin Eq-8 score).
  5. allocate_ranks respects the global budget tau (overshoot by <= one (d_in+d_out) step) and is
     MONOTONIC: more-compact (lower-pi) layers get MORE rank (holding the spectrum fixed). NOTE: this is
     the direction the literal Eq 9-10 produces; it reads inverted vs the task's "more rank to less
     compact" note — see cordapp_init.py header. The check asserts + prints the realized direction.
"""
import sys
import math
import torch

import cordapp_init as CP
import corda_init

torch.manual_seed(0)
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}", flush=True)


# ---------------- 1. compactness closed form ----------------
def rand_orthogonal(n):
    q, r = torch.linalg.qr(torch.randn(n, n))
    return q * torch.sign(torch.diagonal(r)).unsqueeze(0)


spec = torch.tensor([4.0, 1.0, 0.25])
d_out = 5
expected = math.sqrt(d_out * 4.0) / 0.25
C_diag = torch.diag(spec)
Q = rand_orthogonal(3)
C_rot = (Q @ torch.diag(spec) @ Q.T)                      # SPD, same spectrum, rotated
got_diag = CP.compactness(C_diag, d_out)
got_rot = CP.compactness(C_rot, d_out)
got_svals = CP.compactness(C_rot, d_out, svals=torch.linalg.svdvals(C_diag))  # operand override
check("compactness closed form (diagonal SPD)", abs(got_diag - expected) < 1e-4,
      f"got={got_diag:.6f} expected={expected:.6f}")
check("compactness rotation-invariant (SPD spectrum)", abs(got_rot - expected) < 1e-3,
      f"got={got_rot:.6f} expected={expected:.6f}")
check("compactness svals= override", abs(got_svals - expected) < 1e-4,
      f"got={got_svals:.6f} expected={expected:.6f}")


# ---------------- 2. faithful reuse of the static CorDA decomposition ----------------
out, ins, r = 48, 64, 8
W = torch.randn(out, ins)
X = torch.randn(500, ins) @ torch.randn(ins, ins)         # correlated activations -> SPD covariance
C = (X.t() @ X) / 256.0
B, A, Wres = CP.kpm_BAR(W, C, r)
B2, A2, Wres2 = corda_init.corda_kpa_BAR(W, C, r)
dmax = max((B - B2).abs().max(), (A - A2).abs().max(), (Wres - Wres2).abs().max()).item()
check("kpm_BAR == corda_init.corda_kpa_BAR (faithful reuse)", dmax < 1e-4, f"max|diff|={dmax:.2e}")


# ---------------- 3. loss-preserving at per-layer rank & scaling=2 ----------------
def recon_err_at_scaling(W, C, r, scaling):
    B, A, Wres = CP.kpm_BAR(W, C, r)
    B_stored = B / scaling                                # trainer folds scaling into B (B/s)
    eff = Wres + scaling * (B_stored @ A)                 # PEFT effective weight = W_res + s*(B/s @ A)
    return (eff - W).abs().max().item()


for rr in (3, 8, 20):
    e1 = recon_err_at_scaling(W, C, rr, 1.0)
    e2 = recon_err_at_scaling(W, C, rr, 2.0)              # alpha=2r faithful setting
    check(f"KPM loss-preserving (r={rr}, scaling=1)", e1 < 1e-3, f"err={e1:.2e}")
    check(f"KPM loss-preserving (r={rr}, scaling=2 / alpha=2r)", e2 < 1e-3, f"err={e2:.2e}")


# ---------------- 4. dynamic covariance selection picks the most-compact candidate ----------------
# Candidates C_i = c_i * I: sum(sigma)/sigma_max is identical (scalar*I), pi=sqrt(d_out/c_i) decreasing
# in c_i, so argmin score == argmax c_i == smallest pi (most compact). Deterministic.
Wsel = torch.randn(32, 32)
cs = [0.5, 4.0, 1.0, 0.25, 2.0]                           # argmax c = index 1 (c=4.0)
cands = {"m": [c * torch.eye(32) for c in cs]}
weights_sel = {"m": Wsel}
chosen, spectra, info = CP.select_covariances(cands, weights_sel)
pick = info["m"]["chosen_idx"]
check("select_covariances picks most-compact candidate (argmin Eq-8)", pick == cs.index(max(cs)),
      f"chosen_idx={pick} (c={cs[pick]}), expected idx={cs.index(max(cs))} (c={max(cs)})")


# ---------------- 5. rank allocation: budget respect + monotonicity ----------------
# 5 layers, IDENTICAL W (same spectrum ratio) but covariance C_l = c_l*I -> pi decreasing in c_l.
# Isolates pi: allocator should give MORE rank to LOWER-pi (more compact / larger c) layers.
n_layers = 5
c_vals = [0.25, 0.5, 1.0, 2.0, 4.0]                       # ascending c -> descending pi
Wcommon = torch.randn(32, 32)
alloc_cov = {f"L{i}": c_vals[i] * torch.eye(32) for i in range(n_layers)}
alloc_w = {f"L{i}": Wcommon.clone() for i in range(n_layers)}
alloc_spec = {f"L{i}": CP.context_decompose(alloc_w[f"L{i}"], alloc_cov[f"L{i}"])[1]
              for i in range(n_layers)}
fixed_rank = 8
tau = CP.budget_tau(alloc_w, fixed_rank)
ranks, realized, meta = CP.allocate_ranks(alloc_cov, alloc_spec, alloc_w, tau)

step = max(m["d_in"] + m["d_out"] for m in meta.values())
overshoot = realized - tau
# realized must exceed tau (last increment overshot) by no more than one (d_in+d_out) step
budget_ok = (0 < overshoot <= step) and (realized - step <= tau < realized)
check("allocate_ranks respects budget (overshoot <= one step)", budget_ok,
      f"tau={tau} realized={realized} overshoot={overshoot} max_step={step}")

pis = [meta[f"L{i}"]["pi"] for i in range(n_layers)]
rlist = [ranks[f"L{i}"] for i in range(n_layers)]
# c ascending => pi descending => rank should be NON-DECREASING with c (more compact -> more rank),
# and strictly more at the compact extreme than the diffuse extreme.
mono = all(rlist[i] <= rlist[i + 1] for i in range(n_layers - 1)) and rlist[-1] > rlist[0]
check("allocate_ranks monotonic: more-compact (lower pi) -> more rank", mono,
      f"pi={[round(p,2) for p in pis]} ranks={rlist} (c={c_vals})")

# analytic param count == realized (parity sanity)
analytic = sum((meta[n]["d_in"] + meta[n]["d_out"]) * ranks[n] for n in ranks)
check("realized tau == analytic sum_l (d_in+d_out)*r_l", analytic == realized,
      f"analytic={analytic} realized={realized}")


ok = all(results)
print(f"\nCORDA++ CPU VALIDATION: {'PASS' if ok else 'FAIL'} ({sum(results)}/{len(results)} checks)",
      flush=True)
sys.exit(0 if ok else 1)
