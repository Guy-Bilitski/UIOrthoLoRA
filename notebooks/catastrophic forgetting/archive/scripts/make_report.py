"""
Phase-1 deliverables: the adaptation/retention frontier plot + F-delta table +
reproduction-check table + machine-readable verdict inputs. Reads the per-adapter
summaries (results/<run>/summary.json) plus the base reference.

    python make_report.py
"""
import os
import json
import glob

import run_lib

HERE = run_lib.HERE
RES = os.path.join(HERE, "results")

# CLoRA-paper reference numbers (LLaMA-2-7B) for the repro-check table
REF = {
    "base":  {"bbh": 34.91, "mmlu_pro": 18.56, "cs": None},
    "lora":  {"bbh": 26.69, "mmlu_pro": 14.46, "cs": 79.9},
    "clora_k2048": {"bbh": 38.67, "mmlu_pro": 20.59, "cs": 83.7},
}


def load_summaries():
    out = {}
    for p in glob.glob(os.path.join(RES, "*", "summary.json")):
        d = json.load(open(p))
        out[d["run_name"]] = d.get("headline", {})
    return out


def main():
    summaries = load_summaries()
    # base reference (retention only)
    base = {}
    bp = os.path.join(RES, "base_l2-7b", "retention_agg.json")
    bbh_ao = os.path.join(RES, "base_l2-7b_bbhAO", "retention_agg.json")
    if os.path.exists(bp):
        s = json.load(open(bp))["scores"]
        base = {"mmlu_pro": s.get("mmlu_pro")}
    if os.path.exists(bbh_ao):
        base["bbh"] = json.load(open(bbh_ao))["scores"].get("bbh")
    if base:
        base["retention_mean"] = round((base.get("bbh", 0) + base.get("mmlu_pro", 0)) / 2, 2)

    print("\n================ FRONTIER (in-domain CS avg  vs  out-domain retention mean) ================")
    print(f"{'run':28s} {'CS_avg':>7s} {'BBH':>6s} {'MMLU':>6s} {'ret_mean':>8s} {'Fdelta':>7s} "
          f"{'muE':>6s} {'nuD':>6s} {'leak11':>6s} {'offtl':>6s} {'driftU':>6s} {'driftV':>6s}")
    print(f"{'BASE (no adapter)':28s} {'-':>7s} {base.get('bbh','-'):>6} {base.get('mmlu_pro','-'):>6} {base.get('retention_mean','-'):>8}")
    for name in sorted(summaries):
        h = summaries[name]
        g = lambda k: str(h.get(k, '-'))
        print(f"{name:28s} {g('cs_avg'):>7} {g('bbh'):>6} {g('mmlu_pro'):>6} {g('retention_mean'):>8} "
              f"{g('fdelta'):>7} {g('mu_E'):>6} {g('nu_D'):>6} {g('leak11'):>6} {g('offtail_F'):>6} "
              f"{g('drift_U'):>6} {g('drift_V'):>6}")

    print("\n================ REPRODUCTION CHECK (target vs achieved) ================")
    def line(label, achieved, ref):
        if achieved is None:
            return
        print(f"{label:18s} CS {str(achieved.get('cs_avg','-')):>6}/{str(ref.get('cs','-')):>6}  "
              f"BBH {str(achieved.get('bbh','-')):>6}/{ref.get('bbh','-'):>6}  "
              f"MMLU {str(achieved.get('mmlu_pro','-')):>6}/{ref.get('mmlu_pro','-'):>6}")
    print(f"{'base':18s} BBH {base.get('bbh','-')}/{REF['base']['bbh']}  MMLU {base.get('mmlu_pro','-')}/{REF['base']['mmlu_pro']}")
    if "lora_cs_l2-7b_r32" in summaries:
        line("LoRA", summaries["lora_cs_l2-7b_r32"], REF["lora"])
    if "clora_cs_k2048" in summaries:
        line("CLoRA k2048", summaries["clora_cs_k2048"], REF["clora_k2048"])

    # try to draw the frontier plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        groups = {"clora": ("CLoRA", "o", "C0"), "uio": ("UIOrthoLoRA", "s", "C1"),
                  "uilin": ("UILinLoRA", "^", "C2"), "lora": ("LoRA", "*", "C3")}
        for name, h in summaries.items():
            if h.get("cs_avg") is None or h.get("retention_mean") is None:
                continue
            key = ("lora" if name.startswith("lora") else "clora" if name.startswith("clora")
                   else "uilin" if name.startswith("uilin") else "uio")
            lab, mk, col = groups[key]
            ax.scatter(h["cs_avg"], h["retention_mean"], marker=mk, c=col, s=90)
            ax.annotate(name.replace("_cs_l2-7b_r32", "").replace("_kvec410", ""),
                        (h["cs_avg"], h["retention_mean"]), fontsize=6, alpha=0.7)
        if base.get("retention_mean"):
            ax.axhline(base["retention_mean"], ls="--", c="gray", lw=1, label=f"base retention {base['retention_mean']}")
        ax.set_xlabel("in-domain commonsense avg (%)")
        ax.set_ylabel("out-domain retention mean(BBH, MMLU-Pro) (%)")
        ax.set_title("Adaptation / retention frontier (LLaMA-2-7B, commonsense)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        out = os.path.join(RES, "frontier.png")
        fig.tight_layout(); fig.savefig(out, dpi=140)
        print(f"\n[report] frontier plot -> {out}")
    except Exception as e:
        print(f"[report] plot skipped: {e}")


if __name__ == "__main__":
    main()
