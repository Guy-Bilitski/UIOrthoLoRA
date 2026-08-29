# Intruder dimensions — what to write, and what NOT to write

For the paper-writing agent. Everything here is measured on this machine unless marked
PENDING. Status as of 2026-08-29. Numbers update via `python paper_table.py`.

---

## 1. What this experiment adds to the paper

The paper's thesis is that catastrophic forgetting is governed by the **magnitude** of the
weight update. The most prominent competing account in the literature is Shuttleworth et
al. (arXiv 2410.21228), who show that LoRA creates **intruder dimensions** and link them
to forgetting. Xie (arXiv 2607.23711) adds a spectral threshold law for when they appear.

Neither controls for update magnitude. Intruders and magnitude are collinear by
construction — a bigger update creates more and stronger new directions — so a
correlation between intruders and forgetting cannot separate the two.

**Our contribution is the controlled intervention they do not run**: we delete intruder
directions from a trained adapter and compare against controls that remove *exactly the
same amount of update*, differing only in *what* is removed. This is done at each
adapter design's best operating point rather than at an arbitrary learning rate, and it
is measured on retention benchmarks rather than perplexity.

---

## 2. Definitions to state precisely (a reviewer will check these)

**Intruder dimension.** For an adapted matrix `W' = W0 + dW`, take its top-10 left
singular vectors. Direction `u'_j` is an intruder iff

    max_i |<u'_j, u_{0,i}>| < 0.5

over **all** left singular vectors `u_{0,i}` of the pretrained `W0`. This is
Shuttleworth's Definition 3.1 with their main-text window (k = 10) and threshold.

**CRITICAL WORDING.** `W0` is full rank, so its singular vectors span the entire space.
An intruder is therefore **NOT** "a direction outside the pretrained subspace" — that
statement is false and will be caught. The correct phrasing is that it is **not aligned
with any individual pretrained singular direction**, i.e. it is *delocalised* across the
pretrained spectrum. Supporting measurement: 200 random unit directions have
max|cos| ~ 0.059, so **100 % of random directions qualify as intruders at tau = 0.5**.

**Reported statistics.** Intruder fraction (of top-10 slots, aggregated over 160 adapted
matrices per Llama model) and **intruder spectral-energy share**
`sum_{j in I} sigma_j^2 / sum_{j<=10} sigma_j^2`. Energy share is the primary continuous
statistic because the count saturates (99.7 % at F_delta 1.5).

---

## 3. The intervention design

From each trained adapter, seven variants, all evaluated on identical documents:

| arm | what is done | matched to |
|---|---|---|
| A | source, unmodified | — |
| B | delete **all** intruders among the top-10 directions | — |
| C | shrink the **whole** update uniformly to B's magnitude | B (magnitude) |
| D | B rescaled back up to the source magnitude | A (magnitude) |
| E | remove non-intruder content to B's magnitude | B (magnitude) |
| Ep | remove non-intruder content matching B's *removed energy* | B (energy) |
| F | delete an equal **count** of randomly chosen non-intruder directions | B (count) |

Two further arms complete a 2x2 (delete vs keep-only) x (intruder vs non-intruder):
G keeps only the intruder component, H keeps only the non-intruder top-10 component.

**Note for the methods section.** E and Ep cannot delete non-intruder directions outright:
those directions are base-aligned, so removing them from `W'` forces the update to cancel
`W0` and `||dW||` *grows* (measured +13 %). E/Ep therefore act on the update's own
decomposition, `dW_E = (1-a) dW + a P_I dW`. F *does* delete whole directions, which is
the point of the count-matched control; it removes less energy (non-intruders sit deeper,
mean rank 38.9) and `||dW||` grows to 1.069. This is reported, not corrected.

---

## 4. Results (Llama-2-7B; Qwen PENDING — see section 7)

### 4a. Configurations at their best Pareto operating point

| model | method | lr | F_delta | retention | task | intruders | intruder energy |
|---|---|---|---|---|---|---|---|
| Llama-2-7B | LoRA+wd | 5e-4 | 0.395 | 24.87 | 80.00 | 56.8 % | 0.433 |
| Llama-2-7B | CLoRA (k=1024) | 3e-4 | 0.440 | 24.86 | 75.88 | 62.0 % | 0.461 |
| Llama-2-7B | MiLoRA | 3e-4 | 0.558 | 24.28 | 80.00 | 78.4 % | 0.553 |
| *(extra, off-Pareto)* MiLoRA | | 1e-3 | 1.501 | 17.60 | 65.69 | 99.7 % | 0.924 |

Base-model retention = 26.0. Intruder share rises monotonically with update magnitude.
**CLoRA's explicit directional constraint does not suppress intruder formation** — it
sits where its magnitude predicts.

### 4b. The causal test (Llama LoRA+wd; deleting all 908 intruders = 31.2 % of energy)

| arm | F_delta | retention | task |
|---|---|---|---|
| A source | 0.395 | 24.87 | 80.00 |
| C uniform shrink | 0.327 | 25.70 | 61.69 |
| E non-intruder, magnitude-matched | 0.326 | 26.18 | 55.12 |
| **B intruders deleted** | 0.367 | **22.98** | **2.81** |
| **F equal count of non-intruders deleted** | 0.432 | **4.26** | **2.69** |
| D B rescaled to source magnitude | 0.442 | 20.24 | 8.81 |

Contrasts (retention pp / task pp): **B-C** -2.72 / -58.88 · **D-A** -4.63 / -71.19 ·
**B-E** -3.20 / -52.31 · **B-F** +18.72 / +0.12.

### 4c. Magnitude law and residuals

Fitted on the **frozen pool** (which has F_delta and retention for ~60 Llama / ~57 Qwen
adapters but no intruder measurements — its checkpoints were lost in 2026-07):

    Llama:  retention = 18.49 - 6.96 * log F_delta     (R^2 = 0.914, n = 60)
    Qwen:   retention = 11.68 - 14.08 * log F_delta    (R^2 = 0.732, n = 57)

Residuals of our configurations against the Llama law: -0.08, +0.66, +1.76, +1.94, with
intruder energy 0.433, 0.461, 0.553, 0.924. The correlation is **positive** (more
intruder energy -> better-than-predicted retention), i.e. the opposite sign to the
intruder hypothesis. **n = 4 — descriptive only, do not report as a regression result.**

---

## 5. Claims you MAY make

1. **Removing intruder dimensions does not reduce forgetting.** At matched update
   magnitude, retention is unchanged or slightly worse (B-C -2.72 pp, D-A -4.63 pp).
2. **Intruder directions are not pretrained content.** Deleting 908 intruder directions
   costs 18.7 pp less retention than deleting 908 base-aligned directions (22.98 vs
   4.26) — while both destroy the fine-tuned task equally.
3. **Adaptation depends on removing concentrated spectral structure, not on intruder
   identity.** Deleting 908 directions destroys the task whether they are intruders
   (2.81) or not (2.69).
4. **Retention tracks update magnitude**, which the pool law supports at R^2 = 0.914.
5. **A directional constraint (CLoRA) does not suppress intruder formation.**

## 6. Claims you must NOT make

- ❌ "Intruders do not cause forgetting." Too strong. Say: *we find no evidence that
  intruder dimensions are the source of forgetting; at matched update magnitude,
  removing them does not improve retention.*
- ❌ "Intruder dimensions carry the task adaptation." **WITHDRAWN** — arm F refutes it
  (task 2.81 vs 2.69). An earlier draft of our own analysis said this; it is wrong.
- ❌ "Intruders lie outside the pretrained subspace." Mathematically false (see §2).
- ❌ Any claim from the high-magnitude MiLoRA config's arms B/D: both are 0.00/0.00,
  which is a **floor, not a measurement**. At 99.7 % intruder saturation the intervention
  removes the entire top spectrum and destroys the model. Report it only as evidence that
  the intervention has a validity boundary.
- ❌ Any cross-architecture claim until Qwen completes.

## 7. Known limitations to disclose

- **Single seed per configuration.** Three separate seed pathologies are documented in
  this codebase, so single-seed points should be flagged as such.
- **Llama-only so far.** Qwen2.5-7B training NaNs on this machine (H200) though the same
  code, same software versions and same recipe ran cleanly on B200. Under investigation;
  it is a hardware/kernel interaction, not a recipe problem.
- **Evaluation subsets.** Retention = mean(BBH, MMLU-Pro) at 50 documents per subtask;
  task = CS-8 at 200 items per dataset. Validated against the full battery on one
  adapter: 24.87 / 80.00 vs 25.16 / 80.20 — a 0.29 pp difference for a 7x speedup. All
  comparisons are paired (identical documents for every adapter).
- **Arm B is a large, concentrated edit.** It removes 31 % of update energy in one cut,
  which is why arm F (count-matched) and the planned dose-response are needed to
  interpret it.

## 8. Positioning against the literature — EXTENSION, NOT REFUTATION

Frame this as taking the intruder line of work a step further. We do not dispute anything
Shuttleworth et al. or Xie measured, and we reproduce the phenomenon: intruders are
present in every adapter we examined.

| | Shuttleworth et al. (2410.21228) | this work |
|---|---|---|
| comparison | LoRA vs full fine-tuning | across retention-aware LoRA designs |
| LoRA rank | **r = 1 and r = 8** for the causal experiments (r <= 256 for existence) | **r = 32** (deployment scale) |
| operating point | fixed learning rate | each design's best Pareto point |
| intervention | scale the top intruder direction by lambda | delete ALL top-10 intruders **plus magnitude-matched controls (C, D, E, F)** |
| outcome measured | pre-training loss / perplexity | retention benchmarks (BBH, MMLU-Pro) |
| magnitude controlled | no | yes -- this is the addition |

**Why our intruder proportions are so much higher than theirs.** A rank-1 LoRA can
introduce at most one new direction per matrix, so at most 1 of 10 slots can be an
intruder. At r = 32 the update can populate the whole top-10 window, and we measure
57-99 %. We are not contradicting their counts; we are in a regime where the statistic
saturates. Their choice of r = 1 / r = 8 for the causal work is itself evidence that the
metric needs a small update to be discriminative.

**The one correction we make is to interpretation, not to results.** Because
sigma_1(dW) > sigma_1(W0) in *every* matrix we measured, the top of the adapted spectrum
is dominated by the update, so "top-10 direction not aligned with any single pretrained
singular vector" is close to a restatement of "this is the update's own leading
direction". At deployment-scale rank and magnitude the intruder count is therefore
largely definitional, and once magnitude is controlled it carries no additional
information about retention.

**Suggested framing sentence.** *Intruder dimensions are real and ubiquitous, but at
realistic adapter rank they are largely a restatement of where the update lives; the
retention cost of fine-tuning is governed by how far the weights move, not by the
spectral novelty of the directions they move along.*

## 9. Where the numbers live

- `python paper_table.py` — the table, regenerated from `results/`, `--csv` for raw data
- `python magnitude_residuals.py` — pool law + residuals
- `handoff/EXPERIMENT_FINAL.md` — the authoritative protocol
- `handoff/TIERA_RUN_LOG.md` — chronological record, including what was ruled out
- `results/<run>/summary.json` — per-adapter task / retention / F_delta / CE-KL
- `results/intruder/<run>.json` — per-adapter and per-matrix intruder geometry
