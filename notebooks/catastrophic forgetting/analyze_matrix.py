"""
Matrix analyzer (wave-1 commonsense + wave-2 math). Reads campaign_summary.jsonl, keeps
mtx_/mtxm_ runs, aggregates over seeds, and runs THE test:

  retention vs ||dW||_F  -> do all methods COLLAPSE onto one curve (magnitude governs,
                            geometry irrelevant) or SEPARATE (data-aware geometry helps)?
  cs/gsm8k vs ||dW||_F   -> adaptation efficiency (methods ARE allowed to differ here).

F = headline.fdelta (token-weighted ||dW||_F from eval_one_gpu). Usage:
  python analyze_matrix.py          # commonsense (mtx_)
  python analyze_matrix.py math     # math (mtxm_)
"""
import json
import os
import sys
import statistics as st

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
DOMAIN = sys.argv[1] if len(sys.argv) > 1 else "cs"
PREFIX = "mtxm_" if DOMAIN == "math" else "mtx_"


def parse(run):
    """mtx_<method>_<knob>_s<seed> -> (method, knob, seed)."""
    core = run[len(PREFIX):] if run.startswith(PREFIX) else run
    parts = core.split("_")
    seed = parts[-1] if parts[-1].startswith("s") else "s?"
    method = parts[0]
    knob = "_".join(parts[1:-1]) if seed.startswith("s") else "_".join(parts[1:])
    return method, knob, seed


# latest row per run_name
seen = {}
for l in open(os.path.join(RES, "campaign_summary.jsonl")):
    try:
        d = json.loads(l)
    except Exception:
        continue
    if str(d.get("run_name", "")).startswith(PREFIX):
        seen[d["run_name"]] = d

# group by (method, knob) over seeds
groups = {}
_dep = 0
for run, d in seen.items():
    if run.startswith("mtx_sclora"):   # DEPRECATED: buggy input-side normalization; faithful re-run = scl2_*
        _dep += 1
        continue
    m, k, s = parse(run)
    F, ret, cs = d.get("fdelta"), d.get("retention_mean"), d.get("cs_avg")
    if not isinstance(F, (int, float)) or not isinstance(ret, (int, float)):
        continue
    groups.setdefault((m, k), []).append(
        {"F": F, "ret": ret, "cs": cs, "rb": d.get("retention_broad"), "seed": s})


def agg(vs, key):
    xs = [v[key] for v in vs if isinstance(v.get(key), (int, float))]
    if not xs:
        return (None, None)
    return (round(st.mean(xs), 2), round(st.pstdev(xs), 2) if len(xs) > 1 else 0.0)


rows = []
for (m, k), vs in groups.items():
    Fm, Fs = agg(vs, "F"); rm, rs = agg(vs, "ret"); cm, cs_ = agg(vs, "cs"); rbm, _ = agg(vs, "rb")
    rows.append({"method": m, "knob": k, "n": len(vs), "F": Fm, "ret": rm, "ret_sd": rs,
                 "cs": cm, "cs_sd": cs_, "rb": rbm})

if not rows:
    print(f"No {PREFIX}* results yet (matrix still running). Re-run when rows land.")
    sys.exit(0)

rows.sort(key=lambda r: (r["method"], r["F"] if r["F"] is not None else 1e9))
print(f"=== {len(rows)} (method,knob) cells, {sum(r['n'] for r in rows)} runs [{DOMAIN}] "
      f"(skipped {_dep} deprecated mtx_sclora; faithful SC-LoRA = scl2_*) ===")
print(f'{"method":8s} {"knob":7s} {"n":>2s} {"||dW||F":>8s} {"adapt":>6s}±sd {"retain":>7s}±sd {"broad":>6s}')
for r in rows:
    print(f'{r["method"]:8s} {r["knob"]:7s} {r["n"]:>2d} {r["F"] or 0:8.3f} '
          f'{r["cs"] or 0:6.1f}±{r["cs_sd"] or 0:<4.1f} {r["ret"] or 0:7.2f}±{r["ret_sd"] or 0:<4.1f} '
          f'{r["rb"] if r["rb"] is not None else float("nan"):6.2f}')

# THE collapse-vs-separate test: at matched ||dW||_F, how much does retention spread across methods?
print("\n=== COLLAPSE TEST: retention spread across methods at matched ||dW||_F ===")
print("  (spread <~1.5 = noise -> COLLAPSE/magnitude; spread >> noise -> SEPARATION/geometry)")
bins = [(0, 0.15), (0.15, 0.3), (0.3, 0.45), (0.45, 0.6), (0.6, 0.9), (0.9, 1.5), (1.5, 5)]
for lo, hi in bins:
    g = [r for r in rows if r["F"] is not None and lo <= r["F"] < hi]
    if not g:
        continue
    rets = [r["ret"] for r in g]
    spread = round(max(rets) - min(rets), 2)
    bym = ", ".join(sorted({f'{r["method"]}={r["ret"]:.1f}' for r in g}))
    print(f'  F[{lo:.2f},{hi:.2f}): n={len(g):2d} ret {min(rets):.1f}-{max(rets):.1f} spread={spread:4.1f} | {bym}')
