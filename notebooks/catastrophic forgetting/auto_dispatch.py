"""Self-refilling GPU dispatcher — keeps every GPU busy gathering data points.

Problem it solves: with several independent gpu_pool pools, when one pool drains its GPU goes idle
until a human notices. This dispatcher continuously assigns jobs from a master queue to ANY GPU that
is (a) not owned by a still-running gpu_pool and (b) actually idle — so as the existing pools finish,
their GPUs are absorbed automatically and stay saturated until the whole queue is done.

Safety:
- Never touches a GPU whose index appears in a LIVE `gpu_pool.py --gpu_ids ...` process (coexists with
  the running pools; only picks up GPUs after their owning pool exits).
- Never double-runs: skips a job if results/<run>/summary.json exists OR a lock file exists OR the
  dispatcher already launched it (tracks its own child PIDs).
- One job per GPU at a time; a GPU is reused only after its dispatcher child exits.
- Detached, teardown-proof (run via setsid). Idempotent: safe to restart; re-reads locks/results.

Run:
  setsid nice -n 5 .venv/bin/python auto_dispatch.py --jobs jobs/master_dispatch.txt \
      --gpus 0,1,2,3,4,5,6,7 > logs/auto_dispatch.log 2>&1 < /dev/null &
"""
import os, re, sys, time, json, glob, fcntl, argparse, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
LOCKS = os.path.join(HERE, "results", "dispatch_locks")
LOGS = os.path.join(HERE, "logs")
os.makedirs(LOCKS, exist_ok=True); os.makedirs(LOGS, exist_ok=True)
VENV = "/home/guy/UIOrthoLoRA/.venv/bin/python"


def run_name(cmd):
    m = re.search(r"--run_name\s+(\S+)", cmd)
    return m.group(1) if m else None


def done(rn):
    return os.path.exists(os.path.join(HERE, "results", rn, "summary.json"))


def locked(rn):
    return os.path.exists(os.path.join(LOCKS, rn + ".lock"))


def gpus_owned_by_live_pools(my_pid):
    """Set of GPU indices claimed by any live gpu_pool.py process (so we never collide).

    Only count REAL python pool processes (comm == python*): stale `bash -c` launcher
    wrappers keep the full `setsid ... gpu_pool.py --gpu_ids 6,7 ...` text in their
    cmdline long after the pool exited, which froze GPUs as phantom-"pool-owned"
    (GPU6 idled for hours on 2026-07-10 while the watchdog heal-looped)."""
    owned = set()
    try:
        out = subprocess.run(["ps", "-eo", "comm,cmd"], capture_output=True, text=True).stdout
    except Exception:
        return owned
    for line in out.splitlines():
        if "gpu_pool.py" not in line or "grep" in line:
            continue
        if not line.split(None, 1)[0].startswith("python"):
            continue  # bash/sh wrapper, not a live pool
        m = re.search(r"--gpu_ids\s+([0-9,]+)", line)
        if m:
            owned.update(int(x) for x in m.group(1).split(","))
        else:
            m2 = re.search(r"--gpus\s+(\d+)", line)
            if m2:
                owned.update(range(int(m2.group(1))))
    return owned


def gpus_with_compute():
    """GPU indices that currently have ANY compute process."""
    busy = set()
    try:
        uuid2idx = {}
        for l in subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                                capture_output=True, text=True).stdout.splitlines():
            idx, uu = [x.strip() for x in l.split(",")]
            uuid2idx[uu] = int(idx)
        for l in subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"],
                                capture_output=True, text=True).stdout.splitlines():
            uu = l.strip()
            if uu in uuid2idx:
                busy.add(uuid2idx[uu])
    except Exception:
        pass
    return busy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--tag", default="disp")
    ap.add_argument("--poll", type=int, default=90)
    ap.add_argument("--hf_offline", type=int, default=0,
                    help="1 = set HF_HUB_OFFLINE/HF_DATASETS_OFFLINE for children. "
                         "Kills the BBH-init hang class (simultaneous lm-eval inits "
                         "stalling on hub checks); requires all data/models cached — "
                         "an uncached asset then fails FAST (rc!=0 -> requeue) instead "
                         "of hanging silently.")
    a = ap.parse_args()
    my_pid = os.getpid()
    # SINGLE-INSTANCE GUARD: two dispatchers on the same jobs file both picked GPU1
    # within 14s on 2026-07-10 (watchdog relaunch raced the live instance) -> both
    # jobs shared the GPU and one OOM'd. flock held for process lifetime; a second
    # instance exits immediately.
    lockf = open(os.path.join(LOGS, f"dispatcher_{os.path.basename(a.jobs)}.pidlock"), "w")
    try:
        fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print(f"[disp] another dispatcher already owns {a.jobs} — exiting", flush=True)
        return
    lockf.write(f"{my_pid}\n"); lockf.flush()
    all_gpus = [int(x) for x in a.gpus.split(",")]
    with open(a.jobs) as f:
        queue = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    print(f"[disp] {len(queue)} jobs; managing GPUs {all_gpus}; pid {my_pid}", flush=True)

    running = {}      # gpu -> (Popen, run_name)
    launched = 0
    fails = {}        # run_name -> crash count (for bounded requeue)
    while True:
        # reap finished children.
        # CRITICAL: keep the Popen object and use poll(), NOT os.path.exists(/proc/pid).
        # An exited child stays a ZOMBIE until wait()ed, so /proc/<pid> keeps existing and
        # the slot sticks as "busy" forever once all GPUs are occupied (no new Popen ->
        # no interpreter-side reaping). Seen live: d002 GPU4/GPU1 idle with queue remaining.
        for g in list(running):
            p, rn = running[g]
            if p.poll() is not None:
                del running[g]
                print(f"[disp] GPU{g} freed (job {rn} exited rc={p.returncode})", flush=True)
                # FAILURE REQUEUE: a job that died without a summary (OOM, watchdog
                # hang-kill, crash) must release its dispatch lock, or the cell is
                # blocked forever (12 cells silently lost this way on 2026-07-10).
                # Bounded so a deterministic crasher can't loop all night.
                if p.returncode != 0 and not done(rn):
                    fails[rn] = fails.get(rn, 0) + 1
                    if fails[rn] <= 2:
                        try:
                            os.remove(os.path.join(LOCKS, rn + ".lock"))
                            print(f"[disp] {rn} FAILED rc={p.returncode} — lock cleared, retry {fails[rn]}/2", flush=True)
                        except OSError:
                            pass
                    else:
                        print(f"[disp] {rn} PERMANENT-FAIL after 2 retries — lock kept, needs triage", flush=True)
        # remaining work?
        pending = [c for c in queue if not done(run_name(c) or "") and not locked(run_name(c) or "")]
        if not pending and not running:
            print(f"[disp] ALL DONE — {launched} launched, queue empty", flush=True)
            return
        owned = gpus_owned_by_live_pools(my_pid)
        compute = gpus_with_compute()
        for g in all_gpus:
            if g in running:            # we already have a job here
                continue
            if g in owned:              # a live gpu_pool owns this GPU
                continue
            if g in compute:            # something else is using it (settle guard)
                continue
            # pick next undone/unlocked job
            nxt = None
            for c in queue:
                rn = run_name(c)
                if rn and not done(rn) and not locked(rn):
                    nxt = (c, rn); break
            if not nxt:
                break
            cmd, rn = nxt
            open(os.path.join(LOCKS, rn + ".lock"), "w").write(f"gpu{g} {int(time.time())}\n")
            logp = os.path.join(LOGS, f"{a.tag}_{rn}.log")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), PYTHONUNBUFFERED="1",
                       HF_HUB_DISABLE_XET="1", OMP_NUM_THREADS="8", MKL_NUM_THREADS="8")
            if a.hf_offline:
                env["HF_HUB_OFFLINE"] = "1"
                env["HF_DATASETS_OFFLINE"] = "1"
            lf = open(logp, "w"); lf.write(f"# CMD: {cmd}\n# GPU: {g}\n"); lf.flush()
            p = subprocess.Popen(cmd, shell=True, stdout=lf, stderr=subprocess.STDOUT,
                                 cwd=HERE, env=env, executable="/bin/bash")
            running[g] = (p, rn); launched += 1
            print(f"[disp] GPU{g} START {rn} (pid {p.pid})", flush=True)
            time.sleep(3)
        time.sleep(a.poll)


if __name__ == "__main__":
    main()
