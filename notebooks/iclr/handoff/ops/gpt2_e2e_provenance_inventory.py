"""Read-only inventory of the legacy GPT-2 Medium / E2E artifacts under notebooks/E2E (handoff section 8).

Lists every archived score file with its five metrics, whether a run_metadata.json
exists, the recipe fields recoverable from the directory name, and which archived
rows reproduce the manuscript's two UI table rows after rounding. It changes no
historical file, launches nothing and fills no gap: where a row cannot be bound to
a complete run population, that is reported as such.
"""

import csv
import hashlib
import json
import re
from pathlib import Path

E2E = Path("/media/eimtest/data/guyb/UIOrthoLoRA/notebooks/E2E")
OUT = Path("/media/eimtest/data/guyb/UIOrthoLoRA/notebooks/iclr-campaign-20260914/notebooks/iclr/handoff/data/gpt2_e2e_provenance_20260919")
NAME = re.compile(r"^(?P<prefix>.*?)lr_(?P<lr>[0-9.]+)_svalues_(?P<svalues>\d+)_svectors_(?P<svectors>\d+)_seed_(?P<seed>\d+)_init_sigma_(?P<sigma>[0-9.]+)_init_scaler_(?P<scaler>[0-9.]+)$")
PAPER_ROWS = {
    "UIOrthoLoRA": dict(BLEU=68.8, NIST=8.729, METEOR=46.71, ROUGE_L=71.6, CIDEr=2.47, trainable="0.35M", uncertainty="BLEU ±.1, NIST ±.06, METEOR ±.3, ROUGE-L ±.02, CIDEr ±.01"),
    "UILinLoRA": dict(BLEU=68.7, NIST=8.732, METEOR=46.52, ROUGE_L=71.45, CIDEr=2.46, trainable="0.078M", uncertainty="BLEU ±.1, NIST ±.1, METEOR ±.3, ROUGE-L ±.01, CIDEr ±.02"),
}
SCALE = dict(BLEU=100, NIST=1, METEOR=100, ROUGE_L=100, CIDEr=1)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_scores(path):
    scores = {}
    for line in path.read_text().splitlines():
        m = re.match(r"^(BLEU|NIST|METEOR|ROUGE_L|CIDEr):\s*([0-9.]+)", line.strip())
        if m:
            scores[m.group(1)] = float(m.group(2))
    return scores


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    rows = []
    for family, root in (("uiortholora_results", E2E / "outputs/results"), ("lora_results", E2E / "outputs/lora_results")):
        for directory in sorted(p for p in root.iterdir() if p.is_dir()):
            score_file = directory / "scores.txt"
            if not score_file.exists():
                rows.append(dict(family=family, run_directory=directory.name, scores_present=False))
                continue
            scores = parse_scores(score_file)
            metadata = directory / "run_metadata.json"
            recipe = NAME.match(directory.name)
            row = dict(
                family=family,
                run_directory=directory.name,
                scores_present=True,
                scores_sha256=sha(score_file),
                run_metadata_present=metadata.exists(),
                run_metadata_sha256=sha(metadata) if metadata.exists() else "",
                system_outputs_present=(directory / "system_outputs.txt").exists(),
                **{f"score_{k}": scores.get(k) for k in ("BLEU", "NIST", "METEOR", "ROUGE_L", "CIDEr")},
                **({f"name_{k}": v for k, v in recipe.groupdict().items()} if recipe else {}),
            )
            if metadata.exists():
                meta = json.loads(metadata.read_text())
                row.update(meta_model_type=meta.get("model_type"), meta_seed=meta.get("seed"), meta_learning_rate=meta.get("learning_rate"), meta_torch=meta.get("torch"), meta_num_beams=(meta.get("inference_args") or {}).get("num_beams"), meta_target_modules=str((meta.get("peft_config") or {}).get("target_modules")), meta_num_svalues=(meta.get("peft_config") or {}).get("num_svalues_to_adapt"), meta_num_svectors=(meta.get("peft_config") or {}).get("num_svectors_to_adapt"))
            # Which manuscript rows does this single archived file reproduce after the table's rounding?
            matches = []
            for label, paper in PAPER_ROWS.items():
                digits = {"BLEU": 1, "NIST": 3, "METEOR": 2, "ROUGE_L": 1 if label == "UIOrthoLoRA" else 2, "CIDEr": 2}
                if all(k in scores and round(scores[k] * SCALE[k], digits[k]) == paper[k] for k in digits):
                    matches.append(label)
            row["reproduces_paper_row_after_rounding"] = ",".join(matches)
            rows.append(row)
    keys = sorted({k for r in rows for k in r})
    with (OUT / "e2e_result_inventory.csv").open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    scored = [r for r in rows if r.get("scores_present")]
    reproducing = {label: [r["run_directory"] for r in scored if label in r["reproduces_paper_row_after_rounding"].split(",")] for label in PAPER_ROWS}
    summary = dict(
        purpose="gpt2_e2e_provenance_inventory",
        source_tree=str(E2E),
        read_only=True,
        counts=dict(result_directories=len(rows), with_scores=len(scored), with_run_metadata=sum(1 for r in scored if r.get("run_metadata_present")), lora_results=sum(1 for r in scored if r["family"] == "lora_results")),
        paper_rows=PAPER_ROWS,
        archived_files_reproducing_each_paper_row_after_rounding=reproducing,
        seeds_present_by_configuration={},
        finding=(
            "Only single-run score files are archived; run_metadata.json exists for a minority of runs (the later phase1/smart "
            "screens). No archived file records an aggregation over seeds for the two UI rows, so their ±uncertainties cannot be "
            "bound to a run population from these files. Where exactly one archived single run reproduces a row's rounded scores, "
            "that row is consistent with a single-run report; this is a provenance observation, not proof the row is wrong."
        ),
        scripts_present=sorted(p.name for p in E2E.glob("*.py")) + sorted("archive/" + p.name for p in (E2E / "archive").glob("*.py")),
        note="No historical score was modified and no replacement experiment was launched; any correction/removal is a separate author decision.",
    )
    by_config = {}
    for r in scored:
        if r.get("name_svalues"):
            key = f"lr_{r['name_lr']}_svalues_{r['name_svalues']}_svectors_{r['name_svectors']}_sigma_{r['name_sigma']}_scaler_{r['name_scaler']}"
            by_config.setdefault(key, []).append(dict(seed=r["name_seed"], prefix=r.get("name_prefix", ""), BLEU=r["score_BLEU"], directory=r["run_directory"]))
    summary["seeds_present_by_configuration"] = by_config
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(dict(counts=summary["counts"], reproducing=reproducing), indent=1))


if __name__ == "__main__":
    main()
