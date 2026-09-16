"""Launcher: run the FROZEN training pipeline under a non-finite-gradient guard.

train_cs.py is executed UNMODIFIED via runpy (same pattern as run_safe_sdpa.py). This
wrapper only installs two safety hooks on the HF Trainer before the script runs:

  1. optimizer guard  -- before every optimizer step, all gradients are checked for
     finiteness. A non-finite gradient (NaN or inf) means the step is SKIPPED: gradients
     are zeroed, weights and Adam state are left untouched, the run continues. This is
     what torch.amp's GradScaler does for fp16 training and what bf16 training lacks.
     Every skip is logged with its step index; the run ABORTS (exit 3) if more than
     GUARD_MAX_SKIPS steps are skipped (default 640 = 2 % of the run; live-adjustable via
     logs/guard_control.json) or more than GUARD_BURST_MAX skips land inside GUARD_BURST_WINDOW
     consecutive steps, so a run that is genuinely unstable fails fast
     instead of silently training on a diet of skipped batches.
  2. divergence guard -- if the logged training loss exceeds GUARD_LOSS_KILL for
     GUARD_LOSS_CONSEC consecutive logging windows after step GUARD_LOSS_AFTER, the run
     aborts (exit 4). Catches the 2026-08-29 eager failure mode (finite blow-up, loss
     1.2 -> 16) within minutes instead of hours.

Why (2026-09-14): Qwen2.5-7B training on this H200 produced a non-finite gradient in
every attempt (12/12), at a data-order-locked step (seed 43: step 70) under every
attention backend, while the identical recipe trained cleanly on B200 for the frozen
pool. HF Trainer applies a NaN gradient to the weights, so one bad step kills the run.
Skipping that single step is the smallest intervention that lets the pool recipe run
unchanged; the skip count is written to <adapter_dir>/nan_guard.json and must be
reported with the results.

Usage (drop-in, same args as train_cs.py):
  [SAFE_SDPA=default|math|mem_efficient] python run_guarded.py train_cs.py --method lora ...
"""
import os
import sys
import json
import time
import atexit
import runpy
from collections import Counter

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_SKIPS = int(os.environ.get("GUARD_MAX_SKIPS", "640"))     # 2 % of 31,956 steps
BURST_WINDOW = int(os.environ.get("GUARD_BURST_WINDOW", "200"))
BURST_MAX = int(os.environ.get("GUARD_BURST_MAX", "40"))          # >20 % skipped in a window = haywire
CONTROL = os.path.join(HERE, "logs", "guard_control.json")       # {"max_skips": N} re-read on every skip
DUMP_EVERY = 500
LOSS_KILL = float(os.environ.get("GUARD_LOSS_KILL", "3.0"))
LOSS_AFTER = int(os.environ.get("GUARD_LOSS_AFTER", "300"))
LOSS_CONSEC = int(os.environ.get("GUARD_LOSS_CONSEC", "3"))

NAMES = {}          # id(param) -> name, filled when the optimizer is wrapped
LAST = {"loss": None}
STATE = {"steps": 0, "skipped": 0, "skipped_at": [], "bad_loss_streak": 0, "diag": [],
         "aborted": None, "t0": time.time(), "sdpa": os.environ.get("SAFE_SDPA", "default"),
         "max_skips": MAX_SKIPS, "loss_kill": LOSS_KILL, "burst_window": BURST_WINDOW,
         "burst_max": BURST_MAX}


def _current_cap():
    """Cap can be raised/lowered while a run is live by editing logs/guard_control.json."""
    try:
        with open(CONTROL) as f:
            v = int(json.load(f).get("max_skips", STATE["max_skips"]))
        if v != STATE["max_skips"]:
            print(f"[guard] max_skips changed {STATE['max_skips']} -> {v} via {CONTROL}", flush=True)
            STATE["max_skips"] = v
    except Exception:
        pass
    return STATE["max_skips"]


def _argval(flag):
    if flag in sys.argv:
        i = sys.argv.index(flag)
        return sys.argv[i + 1] if i + 1 < len(sys.argv) else None
    return None


def _out_dir():
    root = _argval("--out_root") or "/scratch/cf_models"
    run = _argval("--run_name")
    return os.path.join(root, run) if run else None


def _dump(quiet=False):
    rec = dict(STATE, elapsed_s=round(time.time() - STATE["t0"]))
    if not quiet:
        print(f"[guard] SUMMARY {json.dumps(rec)}", flush=True)
    d = _out_dir()
    if d and os.path.isdir(d):
        with open(os.path.join(d, "nan_guard.json"), "w") as f:
            json.dump(rec, f, indent=1)


atexit.register(_dump)


def _abort(code, why):
    STATE["aborted"] = why
    print(f"[guard] ABORT: {why}", flush=True)
    raise SystemExit(code)


def _grads_finite(opt):
    grads = [p.grad for g in opt.param_groups for p in g["params"] if p.grad is not None]
    if not grads:
        return True
    norms = torch.stack(torch._foreach_norm(grads))
    return bool(torch.isfinite(norms).all().item())


def wrap_optimizer(opt):
    orig_step = opt.step

    def step(closure=None):
        STATE["steps"] += 1
        if not _grads_finite(opt):
            STATE["skipped"] += 1
            STATE["skipped_at"].append(STATE["steps"])
            bad = []
            for g in opt.param_groups:
                for p in g["params"]:
                    if p.grad is None:
                        continue
                    gr = p.grad.detach()
                    n_nan = int(torch.isnan(gr).sum()); n_inf = int(torch.isinf(gr).sum())
                    if n_nan or n_inf:
                        bad.append(f"{NAMES.get(id(p), '?')}[nan={n_nan},inf={n_inf}]")
                    gr.zero_()
            try:
                loss_v = float(LAST["loss"]) if LAST["loss"] is not None else None
            except Exception:
                loss_v = None
            rec = {"step": STATE["steps"], "loss": loss_v, "n_bad": len(bad), "bad": bad[:12]}
            STATE["diag"].append(rec)
            print(f"[guard] diag step {STATE['steps']}: loss={loss_v} non-finite tensors={len(bad)}: "
                  f"{' '.join(bad[:12])}{' ...' if len(bad) > 12 else ''}", flush=True)
            cap = _current_cap()
            recent = sum(1 for k in STATE["skipped_at"] if k > STATE["steps"] - BURST_WINDOW)
            print(f"[guard] non-finite gradient at optimizer step {STATE['steps']} -> step SKIPPED "
                  f"(total skipped {STATE['skipped']}/{cap}; {recent} in last {BURST_WINDOW} steps)",
                  flush=True)
            if STATE["skipped"] > cap:
                _abort(3, f"more than {cap} non-finite steps")
            if recent > BURST_MAX:
                _abort(5, f"{recent} non-finite steps within {BURST_WINDOW} steps")
            return None
        if STATE["steps"] % DUMP_EVERY == 0:
            _dump(quiet=True)
        return orig_step(closure) if closure is not None else orig_step()

    opt.step = step
    opt._guarded = True
    return opt


def install():
    backend = os.environ.get("SAFE_SDPA", "default")
    if backend == "math":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    elif backend == "mem_efficient":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
    elif backend != "default":
        raise SystemExit(f"unknown SAFE_SDPA={backend!r}")
    # cuBLAS may use bf16 (reduced-precision) accumulation across split-K partial sums in
    # bf16 GEMMs. Kernel eligibility depends on pointer alignment, which varies between
    # launches, so a launch can land on a kernel that overflows on particular batches.
    # SAFE_BF16_REDUCE=0 (default) forces fp32 reductions -- strictly more precise, no
    # recipe change. Set SAFE_BF16_REDUCE=1 to keep PyTorch's default.
    if os.environ.get("SAFE_BF16_REDUCE", "0") == "0":
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    STATE["bf16_reduced_reduction"] = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction

    import transformers
    _orig_create = transformers.Trainer.create_optimizer
    _orig_log = transformers.Trainer.log

    def create_optimizer(self, *a, **k):
        out = _orig_create(self, *a, **k)
        opt = self.optimizer
        inner = getattr(opt, "optimizer", opt)          # unwrap accelerate if already wrapped
        if not getattr(inner, "_guarded", False):
            try:
                for n, p in self.model.named_parameters():
                    NAMES[id(p)] = n
            except Exception:
                pass
            wrap_optimizer(inner)
            dt = Counter(str(p.dtype) for g in inner.param_groups for p in g["params"])
            print(f"[guard] optimizer {type(inner).__name__} wrapped; trainable param dtypes "
                  f"{dict(dt)}; sdpa={backend} max_skips={MAX_SKIPS} loss_kill={LOSS_KILL}",
                  flush=True)
        return out

    def log(self, logs, *a, **k):
        loss = logs.get("loss") if isinstance(logs, dict) else None
        step = getattr(getattr(self, "state", None), "global_step", 0)
        if loss is not None and step >= LOSS_AFTER:
            if loss != loss or loss > LOSS_KILL:
                STATE["bad_loss_streak"] += 1
                print(f"[guard] loss {loss} at step {step} exceeds {LOSS_KILL} "
                      f"({STATE['bad_loss_streak']}/{LOSS_CONSEC})", flush=True)
                if STATE["bad_loss_streak"] >= LOSS_CONSEC:
                    _abort(4, f"loss > {LOSS_KILL} for {LOSS_CONSEC} consecutive logs")
            else:
                STATE["bad_loss_streak"] = 0
        return _orig_log(self, logs, *a, **k)

    _orig_ts = transformers.Trainer.training_step

    def training_step(self, *a, **k):
        out = _orig_ts(self, *a, **k)
        try:
            LAST["loss"] = out.detach() if torch.is_tensor(out) else out
        except Exception:
            pass
        return out

    transformers.Trainer.create_optimizer = create_optimizer
    transformers.Trainer.log = log
    transformers.Trainer.training_step = training_step
    print(f"[guard] installed (sdpa={backend}, flash={torch.backends.cuda.flash_sdp_enabled()}, "
          f"mem_eff={torch.backends.cuda.mem_efficient_sdp_enabled()}, "
          f"math={torch.backends.cuda.math_sdp_enabled()}, "
          f"bf16_reduced_precision_reduction={STATE['bf16_reduced_reduction']})", flush=True)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    target = sys.argv[1]
    install()
    sys.argv = [target] + sys.argv[2:]
    runpy.run_path(os.path.join(HERE, target), run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
