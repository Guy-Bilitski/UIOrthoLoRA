"""Qwen 2x2 completion generator (PI: "complete it, but at last" -- queued AFTER all Llama work).

Reconstructs the planned 112-cell Qwen 2x2 (2 domains x 8 arms x 7 LRs, Qwen/Qwen2.5-7B, seed 42;
paper TODO ">=5 adapters on both domains + higher-LR math cells") DIRECTLY from the canonical
template job files -- jobs/auto_qwsw.txt (CS, adapt_task=cs, ret_max_gen 512) and
jobs/auto_qwswm.txt (math, metamathqa_100k, adapt_task=gsm8k, ret_max_gen 256) -- so every emitted
cell uses byte-identical configs to the 55 already-completed qwsw_/qwswm_ cells (internally
consistent arm; the higher-LR math cells lr5e4/lr1e3 are part of this 7-LR grid).

- Resumable: skips any run with results/<run>/summary.json (55 done at assembly: 50 CS + 5 math).
- CorDA arm EMITTED COMMENTED-OUT: PI ruling excludes new old-CorDA cells (CorDA++ replaces CorDA).
  gpu_pool ignores '#' lines; uncomment to restore if the PI wants Qwen-arm internal completeness
  (1 qwsw corda cell is already done, so the Qwen corda column is otherwise a single point).

  python restart_staging/make_qwen_jobs.py --out jobs/frepro4_qwen.txt
"""
import os, re, glob, argparse

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the campaign work-dir
TEMPLATES = ["jobs/auto_qwsw.txt", "jobs/auto_qwswm.txt"]             # CS then math (math runs later)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "restart_staging/frepro4_qwen.txt"))
    a = ap.parse_args()

    done = {os.path.basename(os.path.dirname(p))
            for p in glob.glob(os.path.join(HERE, "results/*/summary.json"))}
    out, n_active, n_corda, n_done = [], 0, 0, 0
    seen = set()
    for tf in TEMPLATES:
        for line in open(os.path.join(HERE, tf)):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = re.search(r"--run_name (\S+)", line)
            run = m.group(1) if m else None
            if run is None or run in seen:
                continue
            seen.add(run)
            if run in done:
                n_done += 1
                continue
            if "_corda_" in run:
                out.append(f"# PI-EXCLUDED (old CorDA; CorDA++ replaces it): {line}")
                n_corda += 1
                continue
            out.append(line)
            n_active += 1
    with open(a.out, "w") as f:
        f.write("\n".join(out) + ("\n" if out else ""))
    print(f"[qwen] planned={len(seen)}  done(skipped)={n_done}  corda(commented)={n_corda}  "
          f"active={n_active} -> {a.out}")


if __name__ == "__main__":
    main()
