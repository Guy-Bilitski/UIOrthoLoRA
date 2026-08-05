# RELEASE_PLAN — code release for the ACL submission (PROPOSAL, nothing executed)

Compiled 2026-07-30 from the release-team audit (anonymization worker report,
completed; CODE_MAP.md, completed). Everything here is a proposal for PI
approval: **no file has been deleted, moved, or modified.** Companion doc
still to be written: RELEASE_README.md (stranger-facing; re-queued after the
session-limit interruption).

## 0. CRITICAL — do this regardless of the release

- `paper/.overleaf-git/.git/config` contains a **live Overleaf git token**
  (`olp_...`) plus the project id and `guyb@sdsai.ai` identity, in plaintext.
  Exclude `paper/.overleaf-git/` from any tar/release, and **rotate the token
  in Overleaf** (Menu -> Git -> regenerate) at your convenience; rotating
  will require updating the local clone's remote URL once.

## 1. Anonymization hit list (must scrub or exclude before release)

| Vector | Where | Proposed handling |
|---|---|---|
| Live Overleaf token + email/name | `paper/.overleaf-git/.git/config` | exclude dir + rotate token |
| Full name + GitHub handle | `handoff/41_EVACUATION_2026-07-17.md:14` (`Guy-Bilitski/UIOrthoLoRA`) | exclude `handoff/` |
| `git config user.email "guyb@sdsai.ai"` | `fleet/collect.sh:28`, `fleet/evacuate.sh:182` | exclude `fleet/` |
| Repo root git remote | `/home/guyb/UIOrthoLoRA/.git/config` | release from a fresh export, never the live checkout |
| **Repo name `UIOrthoLoRA`** (public GitHub, searchable) | 207 files: paths, docs, `--method uiortholora` in `train_cs.py`/`run_lib.py`, `uioW2-4_*` run names | release under a neutral top-level name (`cf-peft-release`); **PI DECISION needed** on the `uiortholora` method string inside tested code (recommended: leave the code untouched, add one README line explaining the internal name) |
| `/home/guy[b]/...` absolute paths (3,225 hits) | `archive/` 2,786; `jobs/` 388; `paper/` 46 (analysis-layer `ROOT=` constants); `fleet/` 19; `scripts/deepseek/` 9; ~15 top-level | mostly solved by exclusions; the analysis layer needs its `ROOT` constants parameterized (env var or relative), a mechanical change proposed for approval |
| Node names `d001`-`d032`, `ubuntu@...` | fleet ops layer + `results/fleet_reg/`, `results/evac_logs/` (59 files, the ONLY leaks inside results/) | exclude those two results/ subtrees; per-run scientific artifacts are clean (full-tree grep) |
| AI-tooling attribution in a heredoc | `sync_d002.sh:22` | excluded with fleet ops |

LaTeX author blocks: all clean (`\author{Anonymous}` everywhere).

## 2. Exclude-from-release directories (keep privately, in git history)

`archive/`, `fleet/` (keep `fleet/evac_merge.py`), `handoff/`, `papers/`
(third-party PDFs, cite instead), `restart_staging/`, `.claude/`,
`paper/.overleaf-git/`, `paper/writing/figures_frozen_backup/`,
`paper/figs_v2/`, `results/fleet_reg/`, `results/evac_logs/`,
`jobs/ce_chunks/` + node-shard job files (keep `jobs/ce_backfill_qwen.txt`
scrubbed, as the paper cites it, plus 1-2 example job lists).

## 3. Archive-candidate files (superseded; propose archive, not delete)

`analyze_magnitude_law.py`, `analyze_matrix.py`, `paper_assets.py`,
`paper_figs_v2.py`, `results_book.py`, `results_book_loop.sh`,
`gpu_watchdog.sh`, `sync_d002.sh`, `evacuate_qwen_adapters.sh`,
`mem_marker.py`, `metamath_prep.py`, `gen_adversarial_jobs.py`,
`rescale_adapters.py` (E1/E3/E6 one-offs; keep as provenance),
`paper/writing/paper_prefreeze_backup_2026-07-18.tex`,
`paper/writing/paper_draft.tex`, `agent_instructions.nd`, legacy
`paper/` root assets. KEEP with documentation: the dated §18 analyzers
(`analyze_full/adversarial/ebatch_2026-07-1*.py`) — they are the only
recompute path for the frozen key_numbers §18; `flag_diverged.py`'s output
`results/quarantine_diverged.txt` stays. `gpu_pool.py`/`auto_dispatch.py`:
PI call (they document "how we ran it"; scrub paths if kept).

## 4. Proposed release tree

See the audit's sketch: neutral root `cf-peft-release/` with `inits/`,
`data_prep/`, `jobs_examples/`, `analysis/` (acl_analysis with parameterized
ROOT), `validation/`, `deepseek/`, `results/` (minus the two leak subtrees),
`docs/` (key_numbers §18-19, frozen exhibits), plus the PEFT fork as a
declared dependency. `portable_parity_pack/` is a good seed but must be
diffed against root first (07-14 snapshot vs 07-17 pipeline files).

## 5. Reproducibility gaps (a stranger today cannot...)

1. ...get the training data: `repro/` is empty on this machine; every config
   points at `repro/LLM-Adapters/ft-training_set/...`. Mitigation exists
   (`portable_parity_pack/fetch_data.py`) but its `data/` holds only
   `.gitkeep`.
2. ...build the environment: no `requirements.txt`/`environment.yml`
   anywhere; the venv builds from the PEFT fork at the repo root, which is
   OUTSIDE this folder and is a hard dependency (must ship or be declared).
3. ...regenerate `tables/table_grand.tex`: generator lost (confirmed);
   inputs intact; ~40-line rewrite. (The compact body version now has a
   committed generator: `acl_analysis/rq1_stats/06_make_grand_compact.py`,
   but it copies from the frozen full table rather than regenerating it.)
4. ...re-evaluate 7B checkpoints: all destroyed in the evacuation; only the
   21 DeepSeek-284B adapters survive (`results/ds_adapters_evac/`).
5. ...run the analysis layer unedited: hardcoded `/home/guyb/...` ROOT
   constants in `acl_analysis/**`, `analysis_a1_a4.py`,
   `analysis_final/op_points_2026-07-17.py`, `make_figs_split_lora_null.py`.
6. ...download base models without instructions (Llama-2 license gate).
7. Referenced-but-absent dirs: `logs/`, `models/`, `results_book/`,
   `jobs/fleet/`, `fleet/ready_nodes.txt`.

## 6. Next steps (pending PI approval)

- [x] Rotate the Overleaf token (done 2026-08-05; bridge updated to the new token, old one invalidated. NOTE: the new token still lives in `paper/.overleaf-git/.git/config` — the directory exclusion below remains mandatory).
- [ ] Approve/adjust the exclusion + archive lists above.
- [ ] Decide the `uiortholora` method-string question (§1).
- [ ] Then: write RELEASE_README.md, add `requirements.txt` freeze,
      parameterize analysis-layer ROOT constants, diff
      portable_parity_pack, rewrite the grand-table generator.
