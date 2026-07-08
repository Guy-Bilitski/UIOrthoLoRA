"""Combine the lean math+cs job files, converting any cell whose adapter already finished
TRAINING (complete safetensors on /scratch) but has no results yet into an EVAL-ONLY job,
so we don't retrain the 7 LoRA+wd cells that were mid-eval when the pool was restarted."""
import json, os, re

done = set()
for l in open("results/campaign_summary.jsonl"):
    l = l.strip()
    if not l:
        continue
    try:
        d = json.loads(l)
    except Exception:
        continue
    if d.get("retention_mean") is not None:
        done.add(d.get("run_name"))


def rn(line):
    m = re.search(r"--run_name (\S+)", line)
    return m.group(1) if m else None


def eval_part(line):
    for seg in line.split(" && "):
        if "eval_one_gpu.py" in seg:
            return seg.strip()
    return line.strip()


out, evalonly, full, skip = [], 0, 0, 0
for f in ["jobs/frepro4_math.txt", "jobs/frepro4_cs.txt"]:
    for line in open(f):
        line = line.rstrip("\n")
        if not line.strip():
            continue
        r = rn(line)
        if r in done:
            skip += 1
            continue
        ad = f"/scratch/cf_models/{r}/adapter_model.safetensors"
        if os.path.exists(ad) and os.path.getsize(ad) > 1_000_000:
            out.append(eval_part(line))
            evalonly += 1
        else:
            out.append(line.strip())
            full += 1

with open("jobs/frepro4_lean.txt", "w") as fh:
    fh.write("\n".join(out) + "\n")
print(f"eval-only (preserved trained): {evalonly}  full train+eval: {full}  "
      f"skipped(done): {skip}  total: {len(out)} -> jobs/frepro4_lean.txt")
