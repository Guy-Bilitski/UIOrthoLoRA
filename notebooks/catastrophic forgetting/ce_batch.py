"""
ce_batch.py -- dispatcher-native batch driver for forgetting_ce.py (CE-to-base forgetting).

Purpose: compute the MiLoRA/Kalajdzievski CE-to-base forgetting metric for MANY saved
adapters as ONE auto_dispatch.py job line, idempotently and concurrency-safely, so the
CE backfill can ride the normal GPU queue without disturbing the training campaign.

Adds on top of forgetting_ce.py (whose model/data/metric code is imported unchanged):
  --glob / --runs / --runs_file   flexible adapter selection (resolved at RUNTIME, so a
                                  late-placed catch-all line picks up adapters trained
                                  after the queue was written)
  skip-done                       any run_name already present in results/forgetting*.jsonl
                                  is skipped; the done-set is REFRESHED before each adapter
                                  so concurrent chunks do not redo each other's work
  per-adapter claim locks         results/ce_locks/<run>.ce.lock created O_EXCL before
                                  processing; a lock with no result older than
                                  --stale_lock_min is treated as stale (crashed chunk)
  line-by-line append             each adapter's result line is appended to --out
                                  immediately (a killed chunk loses at most the adapter
                                  in flight); per-run results/<run>/forgetting.json too
  Llama-only guard                adapters whose adapter_config base_model_name_or_path
                                  is not Llama are skipped (Qwen adapters can't be scored
                                  by this Llama-2 harness)
  --done_marker NAME              on completion writes results/NAME/summary.json
                                  ({"run_name": NAME, "ce_batch": true, "n_done": N})
                                  so auto_dispatch/gpu_watchdog skip-done logic works
  --run_name NAME                 accepted (auto_dispatch parses it from the command
                                  line); defaults --done_marker to the same value

Output schema per line == results/forgetting.jsonl (forgetting_ce, base_entropy,
forgetting_kl, n_positions, run_name, max_length, n_blocks, wall_s, method, adapter_r,
fdelta). NOTE for the table-builder: read results/forgetting*.jsonl (base file + the
per-chunk forgetting_chunk<i>.jsonl files) and dedup by run_name.

Example dispatcher line:
  /home/guy/UIOrthoLoRA/.venv/bin/python ce_batch.py --glob 'frm_*' \
      --out results/forgetting_chunk1.jsonl --max_length 1024 --max_blocks 40 \
      --batch_size 2 --done_marker ce_chunk1 --run_name ce_chunk1
"""
import os
import sys
import json
import glob as globmod
import time
import argparse
import fnmatch
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

RESULTS = os.path.join(HERE, "results")
CE_LOCKS = os.path.join(RESULTS, "ce_locks")


def done_set():
    """run_names already scored, from results/forgetting*.jsonl (cheap; re-read often)."""
    done = set()
    for f in globmod.glob(os.path.join(RESULTS, "forgetting*.jsonl")):
        try:
            with open(f) as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            done.add(json.loads(line)["run_name"])
                        except Exception:
                            pass
        except FileNotFoundError:
            pass
    return done


def is_llama_adapter(adir):
    try:
        base = json.load(open(os.path.join(adir, "adapter_config.json"))).get(
            "base_model_name_or_path", "")
        return "llama" in base.lower()
    except Exception:
        return False


def resolve_candidates(args):
    """Ordered unique run_names matching --runs/--runs_file/--glob with a safetensors file."""
    names = []
    if args.runs:
        names += [r.strip() for r in args.runs.split(",") if r.strip()]
    if args.runs_file:
        with open(args.runs_file) as fh:
            names += [l.strip() for l in fh if l.strip() and not l.startswith("#")]
    if args.glob:
        pats = [p.strip() for p in args.glob.split(",") if p.strip()]
        for d in sorted(os.listdir(args.adapters_root)):
            if any(fnmatch.fnmatch(d, p) for p in pats):
                names.append(d)
    seen, out = set(), []
    for n in names:
        if n in seen:
            continue
        seen.add(n)
        adir = os.path.join(args.adapters_root, n)
        if not os.path.exists(os.path.join(adir, "adapter_model.safetensors")):
            print(f"[list-skip] {n}: no adapter_model.safetensors (not trained yet?)", flush=True)
            continue
        if not is_llama_adapter(adir):
            print(f"[list-skip] {n}: base model is not Llama (Qwen adapter?)", flush=True)
            continue
        out.append(n)
    return out


def claim(run, stale_min):
    """Atomically claim <run>; True if we own it. Stale = lock older than stale_min
    with no result line (crashed chunk) -> re-claim."""
    os.makedirs(CE_LOCKS, exist_ok=True)
    lp = os.path.join(CE_LOCKS, run + ".ce.lock")
    try:
        fd = os.open(lp, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{os.getpid()} {int(time.time())}\n".encode())
        os.close(fd)
        return True
    except FileExistsError:
        try:
            age_min = (time.time() - os.path.getmtime(lp)) / 60.0
        except OSError:
            return False
        if age_min > stale_min:
            print(f"[stale-lock] {run}: lock is {age_min:.0f} min old with no result -> re-claim",
                  flush=True)
            try:
                os.utime(lp, None)  # refresh so only one re-claimer wins the next round
            except OSError:
                pass
            return True
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="", help="comma-separated run names")
    ap.add_argument("--runs_file", default="", help="file with one run name per line")
    ap.add_argument("--glob", default="",
                    help="comma-separated fnmatch patterns over --adapters_root dir names, "
                         "resolved at runtime (e.g. 'frm_*' or '*')")
    ap.add_argument("--out", default="results/forgetting.jsonl",
                    help="jsonl to append result lines to (relative to this dir ok)")
    ap.add_argument("--adapters_root", default="/scratch/cf_models")
    ap.add_argument("--base_model", default="meta-llama/Llama-2-7b-hf")
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--max_blocks", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--hf_home", default=os.environ.get("HF_HOME", "/scratch/hf_cache"))
    ap.add_argument("--done_marker", default="",
                    help="write results/<NAME>/summary.json on completion (dispatcher skip-done)")
    ap.add_argument("--run_name", default="",
                    help="chunk name for auto_dispatch bookkeeping; defaults done_marker to this")
    ap.add_argument("--stale_lock_min", type=float, default=45.0)
    ap.add_argument("--check_base", action="store_true",
                    help="on the FIRST adapter, assert disable_adapter()==fresh base logits")
    ap.add_argument("--dry_run", action="store_true",
                    help="list what would be scored, then exit (no model load)")
    args = ap.parse_args()
    if not args.done_marker and args.run_name:
        args.done_marker = args.run_name

    os.environ.setdefault("HF_HOME", args.hf_home)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    out_path = args.out if os.path.isabs(args.out) else os.path.join(HERE, args.out)

    cand = resolve_candidates(args)
    done0 = done_set()
    todo = [r for r in cand if r not in done0]
    print(f"[plan] {len(cand)} candidates, {len(cand) - len(todo)} already scored, "
          f"{len(todo)} to do -> {out_path}", flush=True)
    if args.dry_run:
        for r in todo:
            print(f"  TODO {r}", flush=True)
        return

    n_done = 0
    if todo:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        import forgetting_ce as fce

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        print(f"[base] loading {args.base_model} ({dtype}) on {device}", flush=True)
        base = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=dtype).to(device).eval()
        blocks = fce.load_wikitext_test_tokens(tokenizer, args.max_length, args.max_blocks,
                                               os.path.join(args.hf_home, "datasets"))
        fresh_for_check = None
        if args.check_base:
            fresh_for_check = AutoModelForCausalLM.from_pretrained(
                args.base_model, dtype=dtype).to(device).eval()

        model = None
        prev = None
        first = True
        for k, run in enumerate(todo):
            if run in done_set():           # refreshed: another chunk beat us to it
                print(f"[skip] {run}: scored by a concurrent chunk", flush=True)
                continue
            if not claim(run, args.stale_lock_min):
                print(f"[skip] {run}: claimed by a concurrent chunk", flush=True)
                continue
            adir = os.path.join(args.adapters_root, run)
            aname = f"a{k}"
            t0 = time.time()
            try:
                if model is None:
                    model = PeftModel.from_pretrained(base, adir, adapter_name=aname).eval()
                else:
                    model.load_adapter(adir, adapter_name=aname)
                    model.set_adapter(aname)
                    if prev is not None:
                        try:
                            model.delete_adapter(prev)
                        except Exception:
                            pass
                prev = aname
                if first and args.check_base and fresh_for_check is not None:
                    d = fce.check_base_matches_fresh(model, fresh_for_check, blocks, device)
                    assert d < 1e-2, f"disable_adapter base mismatch {d:.3e}"
                first = False

                r = fce.forgetting_for_model(model, blocks, args.batch_size, device, dtype)
                r["run_name"] = run
                r["max_length"] = args.max_length
                r["n_blocks"] = int(blocks.shape[0])
                r["wall_s"] = round(time.time() - t0, 1)
                try:
                    cfg = json.load(open(os.path.join(adir, "adapter_config.json")))
                    r["method"] = cfg.get("peft_type")
                    r["adapter_r"] = cfg.get("r")
                except Exception:
                    pass
                sp = os.path.join(RESULTS, run, "summary.json")
                if os.path.exists(sp):
                    try:
                        r["fdelta"] = json.load(open(sp)).get("headline", {}).get("fdelta")
                    except Exception:
                        pass
                # per-adapter durable writes (a killed chunk loses nothing already scored)
                pr = os.path.join(RESULTS, run, "forgetting.json")
                os.makedirs(os.path.dirname(pr), exist_ok=True)
                json.dump(r, open(pr, "w"), indent=2)
                with open(out_path, "a") as fh:
                    fh.write(json.dumps(r) + "\n")
                n_done += 1
                print(f"[{n_done}/{len(todo)}] {run} forgetting_CE={r['forgetting_ce']:.4f} "
                      f"KL={r['forgetting_kl']:.4f} fdelta={r.get('fdelta')} "
                      f"({r['wall_s']}s)", flush=True)
            except Exception:
                print(f"[ERROR] {run}:\n{traceback.format_exc()}", flush=True)
                # release the claim so a later catch-all chunk retries it
                try:
                    os.remove(os.path.join(CE_LOCKS, run + ".ce.lock"))
                except OSError:
                    pass
                continue

    if args.done_marker:
        md = os.path.join(RESULTS, args.done_marker)
        os.makedirs(md, exist_ok=True)
        json.dump({"run_name": args.done_marker, "ce_batch": True, "n_done": n_done},
                  open(os.path.join(md, "summary.json"), "w"), indent=2)
        print(f"[done] marker results/{args.done_marker}/summary.json (n_done={n_done})", flush=True)


if __name__ == "__main__":
    main()
