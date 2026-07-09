"""Shared publication figure style for the magnitude-law paper.

Colorblind-safe categorical palette = the validated light-mode reference palette
from the data-viz skill (worst adjacent CVD dE 24.2, well clear of the >=12
target). Each method has a FIXED colour AND a redundant marker, so identity is
never colour-alone (accessibility non-negotiable). Colour follows the entity, so
the same method is the same colour in every figure.

Import this from a figure script; it only supplies cosmetics + tiny data
helpers. Each figure script reads its own data live.
"""
import os
import re
import json
import matplotlib as mpl

# ---------------------------------------------------------------- paths --------
# scripts live in <WORKDIR>/paper/writing/ ; data lives under <WORKDIR>/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FIGDIR = os.path.join(os.path.dirname(__file__), "figures")
CAMPAIGN = os.path.join(os.path.dirname(__file__), "data", "campaign_summary_clean.jsonl")
GEO_MASTER = os.path.join(ROOT, "results", "geo_drift", "master_labeled.jsonl")
GEO_PERMATRIX = os.path.join(ROOT, "results", "geo_drift", "permatrix")
FORGETTING = os.path.join(ROOT, "results", "forgetting.jsonl")


def summary_fdelta(run):
    """Effective update magnitude F_Delta (CLoRA Eq 3) for a run, from its
    results/<run>/summary.json headline block. None if unavailable."""
    p = os.path.join(ROOT, "results", run, "summary.json")
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p)).get("headline", {}).get("fdelta")
    except Exception:
        return None


# ------------------------------------------------------- categorical palette ---
# canonical display name -> (hex, marker).  Fixed order = palette slot order.
PALETTE = {
    "LoRA":      ("#2a78d6", "o"),   # slot 1 blue
    "LoRA+wd":   ("#1baf7a", "s"),   # slot 2 aqua
    "MiLoRA":    ("#eda100", "^"),   # slot 3 yellow
    "LoRA-Null": ("#008300", "D"),   # slot 4 green
    "CLoRA":     ("#4a3aa7", "v"),   # slot 5 violet
    "SC-LoRA":   ("#e34948", "P"),   # slot 6 red
    "DoRA":      ("#e87ba4", "X"),   # slot 7 magenta
    "CorDA":     ("#eb6834", "<"),   # slot 8 orange
    "PiSSA":     ("#555555", "*"),   # 9th: neutral grey + star (n=1 outlier)
}

# ink / chrome (light surface, print)
INK        = "#0b0b0b"
INK2       = "#52514e"
MUTED      = "#898781"
GRID       = "#e1e0d9"
BASELINE   = "#c3c2b7"
SURFACE    = "#ffffff"
CEILING_C  = "#d03b3b"   # status:critical, reserved for the base-ceiling line
FIT_C      = "#0b0b0b"   # the fitted law curve draws in ink


def color(method):
    return PALETTE.get(method, PALETTE["PiSSA"])[0]


def marker(method):
    return PALETTE.get(method, PALETTE["PiSSA"])[1]


# ------------------------------------------------- run-name -> display method --
_CLASSIFY = [
    (r"lora_null", "LoRA-Null"),
    (r"lorawd",    "LoRA+wd"),
    (r"milora",    "MiLoRA"),
    (r"sclora",    "SC-LoRA"),
    (r"clora",     "CLoRA"),
    (r"corda",     "CorDA"),
    (r"pissa",     "PiSSA"),
    (r"dora",      "DoRA"),
    (r"lora",      "LoRA"),   # last: only matches if none of the above did
]


def method_from_run(run):
    """Map a run_name to its canonical display method (fixes the split()[1]
    labeling bug: lora_null is its own series, wd0 lorawd stays LoRA+wd)."""
    body = run
    # strip a leading campaign prefix so 'lora' in the prefix does not win
    for tag, name in _CLASSIFY:
        if re.search(r"[_/]" + tag + r"[_0-9]", "_" + run):
            return name
    return None


def normalize_method(raw):
    """Normalize a variety of raw method labels to the canonical display name."""
    if raw is None:
        return None
    key = raw.strip().lower().replace("_", "").replace("-", "").replace("+", "")
    table = {
        "lora": "LoRA", "loraplain": "LoRA",
        "lorawd": "LoRA+wd", "loral2": "LoRA+wd",
        "loranull": "LoRA-Null",
        "milora": "MiLoRA", "clora": "CLoRA", "sclora": "SC-LoRA",
        "dora": "DoRA", "corda": "CorDA", "pissa": "PiSSA",
    }
    return table.get(key, raw)


def lr_from_run(run):
    """Parse a learning rate like '..._lr5e4_...' -> 5e-4."""
    m = re.search(r"_lr([0-9])e([0-9])", run)
    if not m:
        return None
    return float(m.group(1) + "e-" + m.group(2))


# ---------------------------------------------------------------- rcParams -----
def apply_rc():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "axes.titleweight": "bold",
        "axes.edgecolor": BASELINE,
        "axes.linewidth": 1.0,
        "axes.labelcolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9.5,
        "legend.frameon": False,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.grid": True,
        "figure.dpi": 110,
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,   # embed TrueType so text stays editable / vectorized
        "ps.fonttype": 42,
    })


def save(fig, stem):
    """Save both a PDF (vector, for the paper) and a PNG (preview) into figures/."""
    os.makedirs(FIGDIR, exist_ok=True)
    pdf = os.path.join(FIGDIR, stem + ".pdf")
    png = os.path.join(FIGDIR, stem + ".png")
    fig.savefig(pdf)
    fig.savefig(png)
    return pdf, png
