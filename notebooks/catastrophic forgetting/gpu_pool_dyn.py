"""
Dynamic single-GPU job pool (2026-09-14, Qwen campaign v2).

Unlike gpu_pool.py (static job list, one job per GPU), this pool POLLS a queue file for
new lines and runs up to --slots jobs concurrently on ONE GPU, starting a new job only
when (a) free GPU memory >= --min_free_mb and (b) --stagger seconds have passed since
the previous start (so the newcomer has finished claiming its memory before the next
one sizes its batches). Starts are serialised with the training daemon through an
fcntl lock on --lock, which the daemon holds while a training process claims memory.

Queue line format:   <run_name>\t<shell command>        ('#' lines ignored)
A job is SKIPPED if results/<run_name>/summary.json already exists when its turn comes
(idempotent re-enqueue). A line consisting of 'STOP' ends the pool once everything
queued before it has finished. A failed job (rc != 0) is retried once after 60 s.
Per-job logs: logs/<tag>_<n>.log ; state: logs/<tag>_state.json
"""
import os, sys, time, json, fcntl, hashlib, argparse, subprocess, threading

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(HERE, "logs")


def free_mb():
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.total,memory.used",
                                   "--format=csv,noheader,nounits"]).decode().strip().split("\n")[0]
    t, u = [int(x) for x in out.split(",")]
    return t - u


def log(msg):
    print(f"[dynpool] {time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}", flush=True)


def control(args):
    """Live-adjustable: logs/dynpool_control.json {slots_training, slots_idle, min_free_mb}."""
    c = {"slots_training": args.slots_training, "slots_idle": args.slots, "min_free_mb": args.min_free_mb}
    try:
        c.update(json.load(open(os.path.join(LOGS, "dynpool_control.json"))))
    except Exception:
        pass
    return c


def training_running():
    return subprocess.call(["pgrep", "-f", "run_guarded.py"], stdout=subprocess.DEVNULL) == 0


def external_pid(run):
    """PID of an already-running eval chain for this run (started by a previous pool)."""
    r = subprocess.run(["pgrep", "-f", "--", f"--run_name {run} --base_model"], capture_output=True, text=True)
    pids = [int(x) for x in r.stdout.split()]
    return pids[0] if pids else None


def read_queue(path):
    items, stop = [], False
    if not os.path.exists(path):
        return items, stop
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            if line.strip() == "STOP":
                stop = True
                continue
            if "\t" not in line:
                continue
            run, cmd = line.split("\t", 1)
            h = hashlib.md5(line.encode()).hexdigest()[:10]
            items.append((h, run.strip(), cmd.strip()))
    return items, stop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--slots", type=int, default=3, help="eval slots when no training runs")
    ap.add_argument("--slots_training", type=int, default=1, help="eval slots while a training runs")
    ap.add_argument("--min_free_mb", type=int, default=36000)
    ap.add_argument("--stagger", type=int, default=150)
    ap.add_argument("--lock", default=os.path.join(LOGS, "gpu_slot.lock"))
    ap.add_argument("--poll", type=int, default=20)
    args = ap.parse_args()
    os.makedirs(LOGS, exist_ok=True)
    # singleton: a second pool instance (e.g. from a campaign relaunch) exits at once
    single = open(os.path.join(LOGS, f"{args.tag}_pool.singleton"), "w")
    try:
        fcntl.flock(single, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("[dynpool] another pool instance holds the singleton lock -> exit", flush=True)
        sys.exit(0)

    state_path = os.path.join(LOGS, f"{args.tag}_state.json")
    state = {"done": {}, "retries": {}, "n": 0}
    if os.path.exists(state_path):
        state.update(json.load(open(state_path)))

    running = {}   # hash -> (Popen, run, idx, t0)

    def save():
        json.dump(state, open(state_path, "w"), indent=1)

    log(f"start: queue={args.queue} slots idle/training={args.slots}/{args.slots_training} min_free={args.min_free_mb}MB stagger={args.stagger}s (live: logs/dynpool_control.json)")
    while True:
        # reap
        for h, (p, run, idx, t0) in list(running.items()):
            if p is None:                      # adopted external chain: poll by pid
                if external_pid(run) is not None:
                    continue
                rc = 0 if os.path.exists(os.path.join(HERE, "results", run, "summary.json")) else 1
            else:
                rc = p.poll()
                if rc is None:
                    continue
            dt = time.time() - t0
            del running[h]
            if rc == 0:
                state["done"][h] = {"run": run, "rc": 0, "s": round(dt)}
                log(f"GPU0 DONE  job{idx} rc=0 {dt:.0f}s {run}")
            else:
                n = state["retries"].get(h, 0)
                if n < 1:
                    state["retries"][h] = n + 1
                    log(f"GPU0 DONE  job{idx} rc={rc} {dt:.0f}s {run} -> will RETRY once")
                else:
                    state["done"][h] = {"run": run, "rc": rc, "s": round(dt)}
                    log(f"GPU0 DONE  job{idx} rc={rc} {dt:.0f}s {run} -> FAILED (no more retries)")
            save()

        items, stop = read_queue(args.queue)
        pending = [(h, r, c) for (h, r, c) in items if h not in state["done"] and h not in running]

        if stop and not pending and not running:
            log("STOP seen and queue drained -> exit")
            break

        c = control(args)
        # logs/train_wanted: the campaign wants to start a training -> stop launching evals
        # beyond the training-time slot count so memory frees up for it
        want = os.path.exists(os.path.join(LOGS, "train_wanted"))
        slots = c["slots_training"] if (training_running() or want) else c["slots_idle"]
        min_free = c["min_free_mb"]
        # adopt chains already running for a pending run (pool restart) instead of duplicating them
        for h, run, cmd in pending:
            if external_pid(run) is not None:
                running[h] = (None, run, -1, time.time())
                log(f"ADOPT running chain for {run} (pid {external_pid(run)})")
        pending = [(h, r, cd) for (h, r, cd) in pending if h not in running]

        if pending and len(running) < slots:
            h, run, cmd = pending[0]
            summ = os.path.join(HERE, "results", run, "summary.json")
            if os.path.exists(summ):
                state["done"][h] = {"run": run, "rc": 0, "s": 0, "skipped": True}
                log(f"SKIP {run} (summary exists)")
                save()
                continue
            fm = free_mb()
            if fm >= min_free:
                with open(args.lock, "w") as lf:
                    fcntl.flock(lf, fcntl.LOCK_EX)
                    fm = free_mb()
                    if fm >= min_free:
                        idx = state["n"]; state["n"] += 1
                        logpath = os.path.join(LOGS, f"{args.tag}_{idx}.log")
                        env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", PYTHONUNBUFFERED="1",
                                   HF_HUB_DISABLE_XET="1", OMP_NUM_THREADS="6", MKL_NUM_THREADS="6")
                        f = open(logpath, "w")
                        f.write(f"# CMD: {cmd}\n# RUN: {run}\n# free_mb_at_start: {fm}\n"); f.flush()
                        p = subprocess.Popen(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT,
                                             cwd=HERE, env=env, executable="/bin/bash")
                        running[h] = (p, run, idx, time.time())
                        save()
                        log(f"GPU0 START job{idx} {run} (free {fm} MB, {len(running)}/{slots} slots{' +training' if slots == c['slots_training'] else ''}) -> {logpath}")
                        time.sleep(args.stagger)      # hold the lock while the job claims memory
                    fcntl.flock(lf, fcntl.LOCK_UN)
                continue
        time.sleep(args.poll)

    fails = [v for v in state["done"].values() if v["rc"] != 0]
    log(f"ALL DONE | {len(state['done'])} jobs | {len(fails)} failures: {[v['run'] for v in fails]}")
    sys.exit(2 if fails else 0)


if __name__ == "__main__":
    main()
