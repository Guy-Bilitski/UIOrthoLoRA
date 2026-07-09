"""Geometry-drift + magnitude battery, phase 2 — CPU, over every saved adapter.

Validated reconstruction (2026-07-09): dW = (alpha/r) * B @ A per target matrix reproduces the
eval-time recorded dw_sv_max exactly for data-aware methods (MiLoRA 48.5074, SC-LoRA 26.9164), so the
residual-save conversion folded the frozen residual into the saved LoRA factors — dW = (alpha/r)*B@A
is correct for ALL methods. Old CorDA (corda_*) is contaminated (exploded dw) and is skipped.

Because dW is low rank (<= r <= 128), all metrics come from the thin factors via two QRs + one r x r
SVD — no full 4096^2 SVD. Phase-1 base-W truncated SVD (results/geo_drift/base_svd/<name>.pt,
top/bottom 256 singular subspaces) supplies the reference subspaces.

Per adapter x matrix we compute:
  fro            = ||dW||_F  (the TRUE Frobenius norm — note the registry `fdelta` is F_delta, NOT this)
  spec           = ||dW||_2  (largest singular value)
  stable_rank    = ||dW||_F^2 / ||dW||_2^2
  eff_rank       = exp(entropy(s/sum s))                 (participation of singular spectrum)
  e_top / e_bot  = fraction of dW's energy whose LEFT singular directions lie in base-W's
                   top-256 / bottom-256 left-singular subspace  (the geometry question:
                   does the update live along W's dominant or minor output directions?)
  align_in_top/bot = same on the INPUT side (right singular vectors vs base-W V_top/V_bot)
  amp_top        = ||U_top^T dW V_top||_F / ||dW||_F     (MiLoRA Table-7 style amplification of the
                   principal block)
Aggregated per adapter (F-weighted mean over matrices + per-target-family breakdown) and written to
results/geo_drift/adapter_metrics.jsonl (one row per adapter). A per-matrix dump goes to
results/geo_drift/permatrix/<run>.jsonl for later per-layer figures.

Run detached, low priority (GPUs are busy with the campaign):
  setsid nice -n 10 .venv/bin/python geo_drift_phase2.py > logs/geo_drift_phase2.log 2>&1 < /dev/null &
"""
import os, json, glob, time, math
import torch
from safetensors import safe_open

torch.set_num_threads(int(os.environ.get("GEO_THREADS", "12")))
HERE = os.path.dirname(os.path.abspath(__file__))
BASE_SVD = os.path.join(HERE, "results", "geo_drift", "base_svd")
OUT_DIR = os.path.join(HERE, "results", "geo_drift")
PM_DIR = os.path.join(OUT_DIR, "permatrix")
os.makedirs(PM_DIR, exist_ok=True)
SCRATCH = "/scratch/cf_models"
TARGETS = ("q_proj", "k_proj", "v_proj", "up_proj", "down_proj")


def base_name(layer, target):
    return f"L{layer}.{target}"


def matrix_metrics(A, B, scaling, bsvd):
    """All metrics for one dW = scaling*B@A via thin factors. A:(r,in) B:(out,r)."""
    Bp = (scaling * B)                       # (out, r)
    # economy QRs of the two thin factors
    Qb, Rb = torch.linalg.qr(Bp, mode="reduced")     # Qb:(out,r) Rb:(r,r)
    Qa, Ra = torch.linalg.qr(A.T, mode="reduced")    # Qa:(in,r)  Ra:(r,r)
    core = Rb @ Ra.T                                   # (r,r); dW = Qb core Qa^T
    Us, s, Vst = torch.linalg.svd(core)               # small r x r
    fro = float(torch.linalg.norm(s))
    if fro < 1e-12:
        return None
    spec = float(s[0])
    p = (s * s); p = p / p.sum()
    eff_rank = float(torch.exp(-(p * (p + 1e-20).log()).sum()))
    stable_rank = float((s * s).sum() / (s[0] ** 2))
    # full singular directions of dW (thin): left = Qb@Us (out,r), right = Qa@Vst^T (in,r)
    Uw = Qb @ Us                                       # (out, r)
    Vw = Qa @ Vst.T                                    # (in, r)
    sd = s                                             # weights
    # energy fractions in base-W subspaces (weight each dW singular dir by s, project onto base U/V)
    def energy_frac(dirs, refU):
        # ||refU^T (dirs * s)||_F^2 / sum s^2 ; dirs:(d,r), refU:(d,k)
        proj = refU.T @ (dirs * sd)                    # (k, r)
        return float((proj * proj).sum() / (sd * sd).sum())
    e_top = energy_frac(Uw, bsvd["U_top"])
    e_bot = energy_frac(Uw, bsvd["U_bot"])
    ein_top = energy_frac(Vw, bsvd["V_top"].T)         # V_top stored (k,in) -> transpose to (in,k)
    ein_bot = energy_frac(Vw, bsvd["V_bot"].T)
    # MiLoRA-style amplification of the principal block: ||U_top^T dW V_top||_F / ||dW||_F
    # dW V_top = (Uw*s) (Vw^T V_top); then U_top^T (...)
    VtopIn = bsvd["V_top"].T                            # (in, k)
    left = bsvd["U_top"].T @ (Uw * sd)                 # (k, r)
    right = Vw.T @ VtopIn                               # (r, k)
    amp_top = float(torch.linalg.norm(left @ right) / fro)
    return dict(fro=fro, spec=spec, stable_rank=stable_rank, eff_rank=eff_rank,
                e_top=e_top, e_bot=e_bot, ein_top=ein_top, ein_bot=ein_bot, amp_top=amp_top)


def process_adapter(run, d):
    cfgp = os.path.join(d, "adapter_config.json")
    if not os.path.exists(cfgp):
        return None
    cfg = json.load(open(cfgp))
    r = cfg.get("r"); alpha = cfg.get("lora_alpha")
    if not r:
        return None
    scaling = alpha / r
    p = os.path.join(d, "adapter_model.safetensors")
    rows = []
    with safe_open(p, "pt") as f:
        keys = [k for k in f.keys() if k.endswith("lora_A.weight")]
        for ak in keys:
            # parse layer + target
            seg = ak.split("layers.")[1]
            layer = int(seg.split(".")[0])
            target = next((t for t in TARGETS if t in ak), None)
            if target is None:
                continue
            bp = os.path.join(BASE_SVD, base_name(layer, target) + ".pt")
            if not os.path.exists(bp):
                continue
            A = f.get_tensor(ak).float()
            B = f.get_tensor(ak.replace("lora_A", "lora_B")).float()
            bsvd = torch.load(bp, map_location="cpu")
            m = matrix_metrics(A, B, scaling, bsvd)
            if m:
                m.update(layer=layer, target=target)
                rows.append(m)
    if not rows:
        return None
    # per-adapter aggregate: F-weighted means (energy/rank), plus max/mean spectral
    W = torch.tensor([x["fro"] for x in rows]); Wsum = float(W.sum())
    def wmean(key):
        return float(sum(x[key] * x["fro"] for x in rows) / Wsum)
    agg = dict(
        run=run, method=cfg.get("_method_hint", run.split("_")[1]),
        r=r, alpha=alpha, n_mat=len(rows),
        fro_total=float(math.sqrt(sum(x["fro"] ** 2 for x in rows))),
        spec_max=max(x["spec"] for x in rows),
        spec_mean=float(sum(x["spec"] for x in rows) / len(rows)),
        stable_rank_w=wmean("stable_rank"), eff_rank_w=wmean("eff_rank"),
        e_top_w=wmean("e_top"), e_bot_w=wmean("e_bot"),
        ein_top_w=wmean("ein_top"), ein_bot_w=wmean("ein_bot"),
        amp_top_w=wmean("amp_top"),
    )
    # dump per-matrix for later per-layer figures
    with open(os.path.join(PM_DIR, run + ".jsonl"), "w") as g:
        for x in rows:
            g.write(json.dumps(x) + "\n")
    return agg


def main():
    t0 = time.time()
    dirs = sorted(glob.glob(os.path.join(SCRATCH, "*")))
    out = os.path.join(OUT_DIR, "adapter_metrics.jsonl")
    done = set()
    if os.path.exists(out):
        for line in open(out):
            try: done.add(json.loads(line)["run"])
            except Exception: pass
    n = 0; skipped = 0
    with open(out, "a") as fout:
        for d in dirs:
            run = os.path.basename(d)
            if run in done:
                continue
            if run.startswith("corda_") or "__" in run:   # contaminated old-CorDA / eval shards
                skipped += 1; continue
            try:
                agg = process_adapter(run, d)
            except Exception as e:
                print(f"[geo2] ERR {run}: {e}", flush=True); continue
            if agg:
                fout.write(json.dumps(agg) + "\n"); fout.flush()
                n += 1
                if n % 20 == 0:
                    print(f"[geo2] {n} adapters  {time.time()-t0:6.0f}s  last={run}", flush=True)
    print(f"[geo2] DONE: {n} adapters ({skipped} skipped) in {time.time()-t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
