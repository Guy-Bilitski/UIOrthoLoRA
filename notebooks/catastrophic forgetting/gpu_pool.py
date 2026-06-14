"""
Tiny GPU job scheduler for the reproduction campaign. Given a list of shell
commands, runs them across the available GPUs (one job per GPU at a time, up to
`--gpus` concurrent). Each command may contain the placeholder `{gpu}` which is
replaced by the assigned GPU index; CUDA_VISIBLE_DEVICES is also exported.

Used to fan training runs and eval shards across the 8 B200s. Jobs are read from
a file (one shell command per line, blank lines / # comments ignored) or passed
inline. Per-job logs go to logs/<tag>_<i>.log.

    python gpu_pool.py --gpus 8 --tag mytag --jobs jobs.txt
"""
import os
import sys
import time
import argparse
import subprocess
import threading
import queue

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(HERE, "logs")


def worker(gpu, jobq, tag, results):
    while True:
        try:
            idx, cmd = jobq.get_nowait()
        except queue.Empty:
            return
        cmd_g = cmd.replace("{gpu}", str(gpu))
        logpath = os.path.join(LOGS, f"{tag}_{idx}.log")
        # cap BLAS threads per job: 128 cores / up-to-8 concurrent jobs -> avoid
        # catastrophic CPU oversubscription (load avg 250+) during SVD/QR/tokenize
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1",
                   HF_HUB_DISABLE_XET="1", OMP_NUM_THREADS="8", MKL_NUM_THREADS="8")
        print(f"[pool] GPU{gpu} START job{idx}: {cmd_g}", flush=True)
        t0 = time.time()
        with open(logpath, "w") as lf:
            lf.write(f"# CMD: {cmd_g}\n# GPU: {gpu}\n")
            lf.flush()
            rc = subprocess.call(cmd_g, shell=True, stdout=lf, stderr=subprocess.STDOUT,
                                 cwd=HERE, env=env, executable="/bin/bash")
        dt = time.time() - t0
        results.append((idx, gpu, rc, round(dt, 1), logpath))
        print(f"[pool] GPU{gpu} DONE  job{idx} rc={rc} {dt:.0f}s -> {logpath}", flush=True)
        jobq.task_done()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--gpu_ids", default="", help="comma list overriding --gpus, e.g. 0,1,2")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--jobs", default="", help="file with one command per line")
    ap.add_argument("--cmd", action="append", default=[], help="inline command (repeatable)")
    args = ap.parse_args()

    os.makedirs(LOGS, exist_ok=True)
    cmds = list(args.cmd)
    if args.jobs:
        with open(args.jobs) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    cmds.append(line)
    if not cmds:
        print("no jobs", file=sys.stderr)
        sys.exit(1)

    gpu_ids = [int(x) for x in args.gpu_ids.split(",")] if args.gpu_ids else list(range(args.gpus))
    jobq = queue.Queue()
    for i, c in enumerate(cmds):
        jobq.put((i, c))

    print(f"[pool] {len(cmds)} jobs across GPUs {gpu_ids} (tag={args.tag})", flush=True)
    results = []
    threads = [threading.Thread(target=worker, args=(g, jobq, args.tag, results)) for g in gpu_ids]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    dt = time.time() - t0

    fails = [r for r in results if r[2] != 0]
    print(f"\n[pool] ALL DONE in {dt:.0f}s | {len(results)} jobs | {len(fails)} failures", flush=True)
    for idx, gpu, rc, d, lp in sorted(results):
        print(f"  job{idx} GPU{gpu} rc={rc} {d}s {lp}", flush=True)
    if fails:
        print("[pool] FAILURES:", [r[0] for r in fails], flush=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
