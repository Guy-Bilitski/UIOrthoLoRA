"""Save an immutable CPU preflight bundle. This command cannot launch training."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from uuid import uuid4

from .artifacts import sha256, utc_now, write_json_new


ROOT=Path(__file__).resolve().parents[3]
HANDOFF=ROOT/"notebooks/iclr/handoff"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-parent",type=Path,default=HANDOFF/"data/campaign_v1/preflight")
    args=parser.parse_args()
    output=args.output_parent/(time.strftime("%Y%m%dT%H%M%SZ",time.gmtime())+"_"+uuid4().hex[:8])
    output.mkdir(parents=True,exist_ok=False)
    env={**os.environ,"CUDA_VISIBLE_DEVICES":"","OMP_NUM_THREADS":"2","MKL_NUM_THREADS":"2",
         "PYTHONPATH":str(ROOT/"src"),"HF_HUB_OFFLINE":"1","HF_DATASETS_OFFLINE":"1"}
    checks=[([sys.executable,"scripts/build_mixing_tables.py","--check"],HANDOFF),
            ([sys.executable,"scripts/check_spectral_algebra.py"],HANDOFF),
            ([sys.executable,"scripts/check_review_additions.py"],HANDOFF),
            ([sys.executable,"scripts/audit_manuscript.py"],HANDOFF),
            (["sha256sum","--check","notebooks/iclr/handoff/GPU_SOURCE_FINGERPRINTS.sha256"],ROOT),
            ([sys.executable,"-m","pytest","-q","notebooks/iclr/campaign/tests","-o","addopts=",
              "--junitxml="+str(output/"tests.xml")],ROOT)]
    report=dict(started_utc=utc_now(),hostname=platform.node(),python=sys.version,
                python_executable=sys.executable,training_launched=False,
                source_revision=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
                source_files={str(p.relative_to(ROOT)):sha256(p) for p in sorted((ROOT/"notebooks/iclr/campaign").rglob("*.py"))},
                packages={d.metadata["Name"]:d.version for d in importlib.metadata.distributions()},checks=[])
    write_json_new(output/"manifest.json",report)
    started=time.perf_counter()
    for i,(command,cwd) in enumerate(checks):
        before=time.perf_counter()
        proc=subprocess.run(command,cwd=cwd,env=env,text=True,capture_output=True)
        record=dict(command=command,cwd=str(cwd),exit_code=proc.returncode,
                    seconds=time.perf_counter()-before,stdout=proc.stdout,stderr=proc.stderr)
        write_json_new(output/f"check_{i:02d}.json",record)
        report["checks"].append(record)
        print(f"{'PASS' if proc.returncode==0 else 'FAIL'}: {' '.join(command[1:])}",flush=True)
    report.update(ended_utc=utc_now(),elapsed_seconds=time.perf_counter()-started,
                  all_checks_passed=all(c["exit_code"]==0 for c in report["checks"]))
    write_json_new(output/"report.json",report)
    print(f"Immutable CPU report: {output}",flush=True)
    return 0 if report["all_checks_passed"] else 1


if __name__=="__main__":
    raise SystemExit(main())
