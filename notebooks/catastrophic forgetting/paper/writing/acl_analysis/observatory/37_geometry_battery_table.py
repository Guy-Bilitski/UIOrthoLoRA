#!/usr/bin/env python3
"""Emit tables/table_geometry_battery.tex: the body geometry table
(label tab:geometry-battery; body Table 2 after the 2026-08-06 redesign
review, replacing the retired tab:fingerprint).

One full-width block: per-method run-level medians of the geometry battery
(stable rank, e_top, ein_top, ein_bot of the trained update), computed
separately for Llama-2-7B and Qwen2.5-7B over on-pool runs. Each method's
advertised-design signature cells (MiLoRA ein_bot, SC-LoRA ein_top,
LoRA-Null e_top, PiSSA e_top+ein_top, matching the body RQ2 prose) are
shaded so the reader sees each design lands in the subspace its
own paper targets.

POOLING: run-level medians per method x model, the same pooling as
30_m3_geometry.py (on_pool rows of m3_master.csv). The frozen reference in
analysis_final/dyn4_geometry.txt section G1 pooled medians over swept cells
instead; its stable-rank medians for LoRA and LoRA{+}wd differ from the
run-level values by 0.5 to 1.9 (Llama LoRA 6.5 vs 4.6, Llama LoRA{+}wd 10.5
vs 9.9, Qwen LoRA 7.1 vs 6.3, Qwen LoRA{+}wd 7.4 vs 6.8); all energy
fractions agree within 0.02. We print the run-level values for consistency
with the body fingerprint table and cross-check against G1, aborting on any
deviation beyond tolerance that is not in the documented list above.

Input-side energies (ein_top_w, ein_bot_w) are not carried in m3_master.csv,
so they are joined in from acl_analysis/insights/pool.csv (same run ids;
the join must cover every on-pool run or the script aborts).

Shading is a single-hue sequential ramp (tints of geoblue, the body table's
blue), value printed in every shaded cell, max tint capped at 42% so black
text keeps contrast; only lightness varies, so grayscale- and CVD-safe.

Requires \\usepackage{colortbl} in the paper preamble (xcolor already
loaded). Pure stdlib. Usage: python3 37_geometry_battery_table.py
"""
import csv
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WRITING = os.path.dirname(os.path.dirname(HERE))          # paper/writing
MASTER = os.path.join(HERE, "m3_master.csv")
POOL = os.path.join(WRITING, "acl_analysis", "insights", "pool.csv")
OUT = os.path.join(WRITING, "tables", "table_geometry_battery.tex")

LLAMA, QWEN = "Llama-2-7B", "Qwen-2.5-7B"          # data keys in m3_master.csv
METHODS = ["lora", "lorawd", "dora", "clora", "milora", "lora_null",
           "sclora", "pissa"]
METHOD_DISPLAY = {
    "lora": "LoRA", "lorawd": "LoRA{+}wd", "dora": "DoRA", "clora": "CLoRA",
    "milora": "MiLoRA", "lora_null": "LoRA-Null", "sclora": "SC-LoRA",
    "pissa": "PiSSA",
}
# Advertised-design signature cell per method (column key in METRICS).
# Closest base-weight-basis proxy for the target each method's own paper
# states, verified against the init code (2026-08-06 review pass):
# sclora_init.py sets B = Q_r, the top eigenvectors of the OUTPUT second
# moment, so SC-LoRA's proxy is e_top (its large ein_top is a mechanical
# side effect of A = Q_r^T W0 and is disclosed, not marked);
# lora_null_init.py puts A's rows in the least-activated INPUT directions,
# so LoRA-Null's proxy is ein_bot; MiLoRA ein_bot; PiSSA both principal
# sides; CLoRA's penalized subspace is random per module (train_cs.py
# CLoRARegularizer) and is not measurable in this basis, so no marker.
SIGNATURE = {"milora": ("eib",), "sclora": ("et",),
             "lora_null": ("eib",), "pissa": ("et", "eit")}
METRICS = ["sr", "et", "eit", "eib"]

# Frozen anchors: run-level medians (n, stable_rank, e_top, ein_top, ein_bot)
# recomputed 2026-08-06 from m3_master.csv (on-pool) joined with
# insights/pool.csv. The script recomputes and ABORTS on any drift.
FROZEN = {
    (LLAMA, "lora"):      (65, 4.6268, 0.0683, 0.0747, 0.0500),
    (LLAMA, "lorawd"):    (274, 9.9391, 0.0728, 0.0850, 0.0492),
    (LLAMA, "dora"):      (52, 3.5313, 0.0703, 0.0893, 0.0500),
    (LLAMA, "clora"):     (79, 7.6141, 0.0609, 0.0665, 0.0495),
    (LLAMA, "milora"):    (109, 5.9303, 0.0673, 0.0726, 0.0919),
    (LLAMA, "lora_null"): (55, 8.7664, 0.1035, 0.1044, 0.0592),
    (LLAMA, "sclora"):    (69, 14.1685, 0.0863, 0.3174, 0.0203),
    (LLAMA, "pissa"):     (5, 18.0460, 0.2210, 0.4571, 0.0277),
    (QWEN, "lora"):       (69, 6.3033, 0.1144, 0.0671, 0.0493),
    (QWEN, "lorawd"):     (47, 6.7940, 0.1164, 0.0746, 0.0472),
    (QWEN, "dora"):       (24, 5.1443, 0.1146, 0.0706, 0.0488),
    (QWEN, "clora"):      (43, 8.5789, 0.0536, 0.0600, 0.0466),
    (QWEN, "milora"):     (48, 10.7041, 0.0690, 0.0609, 0.2142),
    (QWEN, "lora_null"):  (41, 7.8572, 0.1762, 0.0814, 0.1045),
    (QWEN, "sclora"):     (43, 15.2905, 0.1467, 0.2473, 0.0281),
    # no (QWEN, "pissa") arm: printed as --
}

# Cross-check reference: dyn4_geometry.txt G1 (medians over swept cells;
# e_top, ein_top, ein_bot, stable rank). G1 has no lora_null row and no
# Qwen pissa row.
G1 = {
    (LLAMA, "lora"):   (0.072, 0.089, 0.052, 6.5),
    (LLAMA, "lorawd"): (0.073, 0.085, 0.049, 10.5),
    (LLAMA, "dora"):   (0.069, 0.086, 0.050, 3.5),
    (LLAMA, "clora"):  (0.061, 0.066, 0.050, 7.6),
    (LLAMA, "milora"): (0.067, 0.072, 0.106, 5.9),
    (LLAMA, "sclora"): (0.086, 0.318, 0.020, 14.2),
    (LLAMA, "pissa"):  (0.221, 0.457, 0.028, 18.0),
    (QWEN, "lora"):    (0.121, 0.071, 0.053, 7.1),
    (QWEN, "lorawd"):  (0.116, 0.074, 0.047, 7.4),
    (QWEN, "dora"):    (0.115, 0.071, 0.049, 5.1),
    (QWEN, "clora"):   (0.054, 0.060, 0.047, 8.6),
    (QWEN, "milora"):  (0.077, 0.061, 0.195, 10.7),
    (QWEN, "sclora"):  (0.149, 0.260, 0.028, 15.6),
}
# G1 pooled over swept cells, not runs; these four stable-rank medians are
# known to differ beyond the 0.5 tolerance (documented in the header above).
KNOWN_G1_DEVIATIONS = {
    (LLAMA, "lora", "sr"), (LLAMA, "lorawd", "sr"),
    (QWEN, "lora", "sr"), (QWEN, "lorawd", "sr"),
}
G1_TOL = {"sr": 0.5, "et": 0.02, "eit": 0.02, "eib": 0.02}

# Tint anchored per column and model at plain LoRA's value (the reference
# row): there is no single direction-neutral constant, because the 256-cut
# is applied to matrices of different sizes (Qwen k/v have 512 output
# directions), so within-column contrast against LoRA is the honest scale.
MAX_TINT = 42   # cap so black text stays readable on the shaded cell
MIN_TINT = 12   # floor so every signature marker is visible


def tint(value, lo, hi):
    frac = 0.0 if hi <= lo else (value - lo) / (hi - lo)
    return max(MIN_TINT, min(MAX_TINT, round(frac * MAX_TINT)))


def main():
    ein = {}
    with open(POOL) as fh:
        for row in csv.DictReader(fh):
            if row["ein_top_w"] and row["ein_bot_w"]:
                ein[row["rn"]] = (float(row["ein_top_w"]),
                                  float(row["ein_bot_w"]))

    acc = {}
    with open(MASTER) as fh:
        for row in csv.DictReader(fh):
            if (row["on_pool"] != "True" or not row["stable_rank_w"]
                    or row["method"] not in METHODS):
                continue
            if row["run"] not in ein:
                sys.exit(f"JOIN FAILURE: on-pool run {row['run']} has no "
                         f"ein_top_w/ein_bot_w in insights/pool.csv")
            d = acc.setdefault((row["model"], row["method"]),
                               {m: [] for m in METRICS})
            d["sr"].append(float(row["stable_rank_w"]))
            d["et"].append(float(row["e_top_w"]))
            d["eit"].append(ein[row["run"]][0])
            d["eib"].append(ein[row["run"]][1])

    med, n_runs = {}, {}
    for key, d in acc.items():
        med[key] = {m: statistics.median(d[m]) for m in METRICS}
        n_runs[key] = len(d["sr"])

    # ---- hard anchors: abort on drift from the frozen recompute ----
    if set(med) != set(FROZEN):
        sys.exit(f"ANCHOR MISMATCH: method x model set changed: "
                 f"{sorted(set(med) ^ set(FROZEN))}")
    for key, (n, *vals) in FROZEN.items():
        if n_runs[key] != n:
            sys.exit(f"ANCHOR MISMATCH n[{key}]: got {n_runs[key]}, frozen {n}")
        for m, frozen in zip(METRICS, vals):
            got = med[key][m]
            if abs(got - frozen) > 0.00005001:  # 4-dp rounding slack
                sys.exit(f"ANCHOR MISMATCH {m}[{key}]: got {got:.6f}, "
                         f"frozen {frozen}")

    # ---- soft cross-check vs G1 (different pooling; known deviations) ----
    for key, (g_et, g_eit, g_eib, g_sr) in G1.items():
        for m, ref in zip(["et", "eit", "eib", "sr"],
                          [g_et, g_eit, g_eib, g_sr]):
            diff = abs(med[key][m] - ref)
            if diff > G1_TOL[m]:
                if (key[0], key[1], m) in KNOWN_G1_DEVIATIONS:
                    print(f"[g1] known pooling deviation {m}{key}: "
                          f"run-level {med[key][m]:.3f} vs G1 {ref}")
                else:
                    sys.exit(f"G1 CROSS-CHECK FAILURE {m}{key}: "
                             f"run-level {med[key][m]:.4f} vs G1 {ref} "
                             f"(tol {G1_TOL[m]}, not a documented deviation)")

    def cell(key, m):
        if key not in med:
            return "--"
        val = med[key][m]
        txt = f"{val:.1f}" if m == "sr" else f"{val:.3f}"
        if m in SIGNATURE.get(key[1], ()):
            lora_ref = med[(key[0], "lora")][m]
            col_hi = max(med[k][m] for k in med if k[0] == key[0])
            return f"\\cellcolor{{geoblue!{tint(val, lora_ref, col_hi)}}}{txt}"
        return txt

    lines = []
    a = lines.append
    a("% AUTO-GENERATED by acl_analysis/observatory/37_geometry_battery_table.py")
    a("% from m3_master.csv (on-pool; 30_m3_geometry.py run-level pooling) joined")
    a("% with acl_analysis/insights/pool.csv for the input-side energies, and")
    a("% cross-checked against analysis_final/dyn4_geometry.txt G1 (which pooled")
    a("% over swept cells; its LoRA/LoRA+wd stable-rank medians differ by 0.5-1.9,")
    a("% all energies agree within 0.02). The generator aborts on any drift from")
    a("% its frozen anchors. Do not hand-edit; rerun the script.")
    a("% Needs \\usepackage{colortbl} (xcolor already in the preamble); geoblue")
    a("% is also defined by tables/table_geometry_fingerprint.tex.")
    a("\\providecolor{geoblue}{RGB}{0,114,178}")
    a("\\begin{table*}[t]")
    a("\\centering")
    a("\\footnotesize")
    a("\\setlength{\\tabcolsep}{5pt}")
    a("\\renewcommand{\\arraystretch}{1.12}")
    a("\\begin{tabular}{l rrrrr rrrrr}")
    a("\\toprule")
    a(" & \\multicolumn{5}{c}{Llama-2-7B} & \\multicolumn{5}{c}{Qwen2.5-7B} \\\\")
    a("\\cmidrule(lr){2-6} \\cmidrule(lr){7-11}")
    a(" & & & \\multicolumn{3}{c}{share of update energy in} & & &"
      " \\multicolumn{3}{c}{share of update energy in} \\\\")
    a(" & & & \\multicolumn{3}{c}{base directions} & & &"
      " \\multicolumn{3}{c}{base directions} \\\\")
    a("\\cmidrule(lr){4-6} \\cmidrule(lr){9-11}")
    a("Method & $n$ & SR & top out & top in & bottom in"
      " & $n$ & SR & top out & top in & bottom in \\\\")
    a("\\midrule")
    NO_TARGET = ["lora", "lorawd", "dora", "clora"]
    DECLARED = ["milora", "lora_null", "sclora", "pissa"]
    def emit_row(meth):
        cells = [METHOD_DISPLAY[meth]]
        for model in (LLAMA, QWEN):
            key = (model, meth)
            cells.append(str(n_runs[key]) if key in med else "--")
            cells.extend(cell(key, m) for m in METRICS)
        a(" & ".join(cells) + " \\\\")
    a("\\multicolumn{11}{l}{\\emph{states no target subspace"
      " (LoRA row is the reference for every column)}} \\\\")
    for meth in NO_TARGET:
        emit_row(meth)
    a("\\midrule")
    a("\\multicolumn{11}{l}{\\emph{states a target subspace"
      " (marked cell is the closest proxy in this basis)}} \\\\")
    for meth in DECLARED:
        emit_row(meth)
    a("\\bottomrule")
    a("\\end{tabular}")
    a("\\caption{Update geometry per method and base model: run-level")
    a("medians over the on-pool 7B runs, with plain LoRA as the reference row")
    a("for every column and the number of runs per block. Each energy column")
    a("is the share of the update's energy in a fixed $256$ base-weight")
    a("singular directions (top output, top input, bottom input); the neutral")
    a("level differs by column and model (Qwen's attention k/v matrices have")
    a("only $512$ output directions), so values read against the column's")
    a("LoRA reference, not across columns. The marked cell in each row is the")
    a("closest proxy in this basis for the target the method's own paper")
    a("states, and it sits above the LoRA reference in every case; SC-LoRA's")
    a("large top-input share is an initialization side effect, and CLoRA")
    a("penalizes a random per-module subspace this basis does not measure. SR")
    a("is the stable rank of the update; the stable-rank split recurs at 284B")
    a("(Table~\\ref{tab:ds284b}); PiSSA has no Qwen-2.5 arm.}")
    a("\\label{tab:geometry-battery}")
    a("\\end{table*}")

    with open(OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {OUT}")
    for model in (LLAMA, QWEN):
        print(model, {METHOD_DISPLAY[m]: round(med[(model, m)]["sr"], 1)
                      for m in METHODS if (model, m) in med})


if __name__ == "__main__":
    main()
