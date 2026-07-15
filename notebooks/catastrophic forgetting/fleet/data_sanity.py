#!/usr/bin/env python3
"""Data-health check for the CF campaign — inspect the CONTENT of summary.json cells,
not just that files exist. Catches the "GPUs pegged but output is garbage/degenerate"
failure mode (an eval that crashed mid-way and wrote zeros/NaNs on an otherwise-fine
adapter), while NOT crying wolf over the two benign categories:

  * DIVERGED (expected): extreme-LR sweep endpoints where the update magnitude blew up
    (fdelta NaN, or huge finite fdelta) and retention collapsed to the chance floor.
    This is real data at the high-magnitude end of the axis — the paper's own thesis —
    not a pipeline fault. (NaN cells still can't feed a Pearson fit; consolidation drops
    them, but that is a fit-time concern, not a health alarm.)
  * OLD-SCHEMA (benign): pre-campaign exploratory cells (e.g. uioW3_*, valfix_*) that use
    a different headline schema and predate the seed-fill run.

Classification per adapter cell (base-only rows are checked only for NaN):
  ALARM  -> retention_mean/broad is 0 or NaN WHILE fdelta is in a normal healthy band
            (0 < fdelta <= FDELTA_DIVERGE): a good adapter with a broken eval. The real bug.
  INFO   -> diverged (fdelta NaN or > FDELTA_DIVERGE) with degenerate retention: expected.
  INFO   -> old-schema (no adapt_task in headline, or uio-specific keys present): benign.

Usage:
  python3 fleet/data_sanity.py                 # scan all local results
  python3 fleet/data_sanity.py --since_min 30  # only cells modified in the last 30 min
  python3 fleet/data_sanity.py --show_info      # also list the INFO (benign/expected) cells
Exit code: 2 if any ALARM, else 0.
"""
import argparse, glob, json, math, os, time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(HERE, "results")

FDELTA_DIVERGE = 5.0   # fdelta above this = update magnitude blew up (healthy LoRA is < ~2)
_UIO_KEYS = ("mu_E", "nu_D", "leak11", "offtail_F", "drift_U")  # old exploratory schema markers


def _finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


def _finite_pos(x):
    return _finite(x) and x > 0


def _is_base_only(s):
    if not s.get("adapter"):
        return True
    fd = s.get("fdelta") or {}
    return fd.get("n_matrices", 0) == 0 and (s.get("method", "").lower() in ("base", "none", "unknown"))


def classify(path):
    """Return (severity, msg) where severity in {'ok','ALARM','INFO'}."""
    try:
        s = json.load(open(path))
    except Exception as e:
        return ("ALARM", f"unreadable: {e}")
    head = s.get("headline") or {}
    fd = s.get("fdelta") or {}

    # old exploratory schema -> benign
    if any(k in head for k in _UIO_KEYS):
        return ("INFO", "old-schema (uio exploratory)")
    if _is_base_only(s):
        # base ceilings: only NaN in headline is a problem
        bad = [k for k, v in head.items() if isinstance(v, float) and not math.isfinite(v)]
        return ("ALARM", f"base-only NaN in {bad}") if bad else ("ok", "")
    if "adapt_task" not in head:
        return ("INFO", "old-schema (no adapt_task)")

    fdt = fd.get("fdelta_token_weighted")
    diverged = (fdt is None) or (not math.isfinite(fdt) if isinstance(fdt, float) else False) or (
        _finite(fdt) and fdt > FDELTA_DIVERGE)

    adapt = head.get("adapt_task")
    acc = head.get(adapt) if adapt in head else head.get("cs_avg")
    ret_m = head.get("retention_mean")
    ret_b = head.get("retention_broad")

    ret_degenerate = (ret_m is None) or (not _finite(ret_m)) or (ret_m == 0) \
        or (ret_b is not None and (not _finite(ret_b) or ret_b == 0))
    acc_degenerate = _finite(acc) and not (0 < acc <= 100.0001)

    if diverged:
        # extreme-LR endpoint; degenerate downstream values are expected, not a fault
        tag = "fdelta=NaN" if not _finite(fdt) else f"fdelta={fdt:.1f}"
        return ("INFO", f"diverged ({tag}, retention_mean={ret_m})")

    # fdelta is in a healthy band -> the adapter trained fine. Now degeneracy = real fault.
    if ret_degenerate or acc_degenerate:
        return ("ALARM", f"healthy fdelta={fdt} but adapt_acc={acc}, ret_mean={ret_m}, ret_broad={ret_b} (eval crash?)")
    if not _finite_pos(fd.get("dw_sv_max")) or not _finite_pos(fd.get("dw_sv_mean")):
        return ("ALARM", f"dw_sv non-finite: max={fd.get('dw_sv_max')} mean={fd.get('dw_sv_mean')}")
    return ("ok", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since_min", type=float, default=0, help="only cells modified in the last N min (0=all)")
    ap.add_argument("--limit", type=int, default=25, help="max ALARM cells to print (newest first)")
    ap.add_argument("--show_info", action="store_true", help="also list benign/expected INFO cells")
    args = ap.parse_args()

    paths = glob.glob(os.path.join(RESULTS, "*", "summary.json"))
    now = time.time()
    if args.since_min > 0:
        paths = [p for p in paths if now - os.path.getmtime(p) <= args.since_min * 60]
    paths.sort(key=os.path.getmtime, reverse=True)

    alarms, infos = [], []
    for p in paths:
        sev, msg = classify(p)
        if sev == "ALARM":
            alarms.append((os.path.getmtime(p), os.path.basename(os.path.dirname(p)), msg))
        elif sev == "INFO":
            infos.append((os.path.getmtime(p), os.path.basename(os.path.dirname(p)), msg))

    win = f" (last {args.since_min:g} min)" if args.since_min else ""
    print(f"[data_sanity] scanned {len(paths)} summaries{win}; "
          f"{len(alarms)} ALARM, {len(infos)} INFO(diverged/old-schema), "
          f"{len(paths)-len(alarms)-len(infos)} clean")
    for mt, run, msg in alarms[: args.limit]:
        print(f"  ALARM {run} (age {(now-mt)/60:.0f}m): {msg}")
    if len(alarms) > args.limit:
        print(f"  ... and {len(alarms)-args.limit} more ALARM")
    if args.show_info:
        for mt, run, msg in infos[: args.limit]:
            print(f"  info  {run} (age {(now-mt)/60:.0f}m): {msg}")
    raise SystemExit(2 if alarms else 0)


if __name__ == "__main__":
    main()
