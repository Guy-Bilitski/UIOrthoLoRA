import json
import os

# The content of the notebook cells
cells = []

# =============================================================================
# CELL 1: Markdown Header
# =============================================================================
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# OrthoLoRA Adapter Analysis\n",
        "\n",
        "Comprehensive analysis comparing LoRA, VeRA, and OrthoLoRA adapters across BigBench, MMLU, and Task-Specific evaluations.\n",
        "\n",
        "**Data Source:** `src/analysis/adapters_results/`"
    ]
})

# =============================================================================
# CELL 2: Imports and Style
# =============================================================================
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import matplotlib\n",
        "import seaborn as sns\n",
        "import os\n",
        "import re\n",
        "import warnings\n",
        "warnings.filterwarnings('ignore')\n",
        "\n",
        "# Style Setup\n",
        "plt.rcParams.update({\n",
        "    'figure.figsize': (12, 8),\n",
        "    'font.size': 12,\n",
        "    'axes.titlesize': 14,\n",
        "    'axes.labelsize': 13,\n",
        "    'legend.fontsize': 10,\n",
        "    'figure.dpi': 150,\n",
        "    'savefig.dpi': 150,\n",
        "    'savefig.bbox': 'tight',\n",
        "    'axes.spines.top': False,\n",
        "    'axes.spines.right': False,\n",
        "})\n",
        "\n",
        "ADAPTER_COLORS = {\n",
        "    'lora': '#E63946',\n",
        "    'vera': '#457B9D',\n",
        "    'uiortholora': '#2A9D8F',\n",
        "}\n",
        "ADAPTER_LABELS = {\n",
        "    'lora': 'LoRA',\n",
        "    'vera': 'VeRA',\n",
        "    'uiortholora': 'OrthoLoRA',\n",
        "}\n",
        "SIZE_MARKERS = {\n",
        "    'small': 'o',\n",
        "    'large': 's',\n",
        "}\n",
        "\n",
        "print(\"Libraries loaded and style set.\")"
    ]
})

# =============================================================================
# CELL 3: Configuration & Helpers
# =============================================================================
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Configuration\n",
        "BASE_PATH = 'src/analysis/adapters_results'\n",
        "\n",
        "# Select Dataset and Threshold here\n",
        "DATASET = 'hotpotqa'       # Options: 'hotpotqa', 'triviaqa'\n",
        "THRESHOLD = '0.8'          # Options: '0.6', '0.8'\n",
        "\n",
        "TARGET_DIR = os.path.join(BASE_PATH, DATASET, f'threshold_{THRESHOLD}')\n",
        "print(f\"Analyzing data from: {TARGET_DIR}\")\n",
        "\n",
        "# Helper functions for parsing names\n",
        "def get_adapter_type(name):\n",
        "    name = str(name).lower()\n",
        "    if 'uiortholora' in name: return 'uiortholora'\n",
        "    if 'vera' in name: return 'vera'\n",
        "    if 'lora' in name: return 'lora'\n",
        "    return 'unknown'\n",
        "\n",
        "def extract_lr(name):\n",
        "    m = re.search(r'lr([\\de\\-\\.]+)', str(name))\n",
        "    if m:\n",
        "        return m.group(1)\n",
        "    return None\n",
        "\n",
        "def classify_size(row):\n",
        "    adapter = row['adapter']\n",
        "    name = str(row.get('model_name', ''))\n",
        "    rank = row.get('rank', 0)\n",
        "    \n",
        "    if adapter == 'vera':\n",
        "        return 'small'\n",
        "    elif adapter == 'lora':\n",
        "        try:\n",
        "            return 'large' if float(rank) >= 3 else 'small'\n",
        "        except:\n",
        "            return 'small'\n",
        "    elif adapter == 'uiortholora':\n",
        "        # Parse s and v dimensions if available in name like _s1024_v32\n",
        "        s_match = re.search(r'_s(\\d+)', name)\n",
        "        v_match = re.search(r'_v(\\d+)', name)\n",
        "        if s_match and v_match:\n",
        "            s = int(s_match.group(1))\n",
        "            v = int(v_match.group(1))\n",
        "            return 'small' if s * v <= 1024 * 32 else 'large'\n",
        "        if '_v0_' in name or name.endswith('_v0'):\n",
        "            return 'small'\n",
        "    return 'small'"
    ]
})

# =============================================================================
# CELL 4: Data Loading
# =============================================================================
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Load CSVs\n",
        "try:\n",
        "    # Task specific data (accuracy, negative shift)\n",
        "    gemma_path = os.path.join(TARGET_DIR, 'gemma-12b.csv')\n",
        "    llama_path = os.path.join(TARGET_DIR, 'llama-3b.csv')\n",
        "    \n",
        "    df_gemma = pd.read_csv(gemma_path)\n",
        "    # df_llama = pd.read_csv(llama_path) # Uncomment if comparing architectures\n",
        "    \n",
        "    # Generalization data\n",
        "    bb_path = os.path.join(TARGET_DIR, 'bigbench_summary.csv')\n",
        "    mmlu_path = os.path.join(TARGET_DIR, 'mmlu_summary.csv')\n",
        "    \n",
        "    df_bb = pd.read_csv(bb_path)\n",
        "    df_mmlu = pd.read_csv(mmlu_path)\n",
        "    \n",
        "    print(\"Files loaded successfully.\")\n",
        "    print(f\"Gemma Task Rows: {len(df_gemma)}\")\n",
        "    print(f\"BigBench Rows: {len(df_bb)}\")\n",
        "    print(f\"MMLU Rows: {len(df_mmlu)}\")\n",
        "\n",
        "except FileNotFoundError as e:\n",
        "    print(f\"Error loading files: {e}\")\n",
        "    print(\"Please ensure the directory structure exists and contains the CSV files.\")"
    ]
})

# =============================================================================
# CELL 5: Data Merging & Preprocessing
# =============================================================================
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Normalize columns for merging\n",
        "# Assumption: All CSVs have a 'model_name' or 'adapter_name' column that acts as a unique key\n",
        "\n",
        "def normalize_df(df, prefix=''):\n",
        "    # Ensure we have a join key\n",
        "    if 'model_name' not in df.columns and 'model_path' in df.columns:\n",
        "        df['model_name'] = df['model_path'].apply(lambda x: x.split('/')[-1])\n",
        "    elif 'raw_adapter_name' in df.columns:\n",
        "        df['model_name'] = df['raw_adapter_name']\n",
        "    \n",
        "    # Clean join key\n",
        "    df['join_key'] = df['model_name'].astype(str).str.strip()\n",
        "    return df\n",
        "\n",
        "df_gemma = normalize_df(df_gemma)\n",
        "df_bb = normalize_df(df_bb)\n",
        "df_mmlu = normalize_df(df_mmlu)\n",
        "\n",
        "# Rename metric columns to avoid collisions before merge\n",
        "df_bb_clean = df_bb[['join_key', 'mean_accuracy']].rename(columns={'mean_accuracy': 'bigbench_acc'})\n",
        "df_mmlu_clean = df_mmlu[['join_key', 'mmlu_acc']].rename(columns={'mmlu_acc': 'mmlu_accuracy'})\n",
        "\n",
        "# Merge\n",
        "df_merged = df_gemma.merge(df_bb_clean, on='join_key', how='left')\n",
        "df_merged = df_merged.merge(df_mmlu_clean, on='join_key', how='left')\n",
        "\n",
        "# Feature Engineering\n",
        "df_merged['adapter'] = df_merged['model_name'].apply(get_adapter_type)\n",
        "df_merged['lr'] = df_merged['model_name'].apply(extract_lr)\n",
        "df_merged['size'] = df_merged.apply(classify_size, axis=1)\n",
        "\n",
        "# Ensure numerics\n",
        "numeric_cols = ['accuracy', 'negative_shift_score', 'bigbench_acc', 'mmlu_accuracy']\n",
        "for col in numeric_cols:\n",
        "    if col in df_merged.columns:\n",
        "        df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce')\n",
        "\n",
        "print(f\"Merged Dataset Size: {len(df_merged)}\")\n",
        "df_merged[['adapter', 'lr', 'accuracy', 'negative_shift_score', 'bigbench_acc', 'mmlu_accuracy']].head()"
    ]
})

# =============================================================================
# CELL 6: Figure 1 - Accuracy vs Forgetting
# =============================================================================
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# FIGURE 1: Task Accuracy vs Negative Shift — Adapter Comparison\n",
        "fig, ax = plt.subplots(figsize=(12, 8))\n",
        "\n",
        "for adapter in ['lora', 'vera', 'uiortholora']:\n",
        "    subset = df_merged[df_merged['adapter'] == adapter]\n",
        "    subset = subset[subset['negative_shift_score'] > 0]  # Only valid shift scores\n",
        "    ax.scatter(\n",
        "        subset['negative_shift_score'],\n",
        "        subset['accuracy'],\n",
        "        c=ADAPTER_COLORS.get(adapter, 'gray'),\n",
        "        label=ADAPTER_LABELS.get(adapter, adapter),\n",
        "        alpha=0.7,\n",
        "        s=80,\n",
        "        edgecolors='white',\n",
        "        linewidths=0.5,\n",
        "        zorder=3\n",
        "    )\n",
        "\n",
        "ax.set_xlabel('Negative Shift Score (lower = less forgetting)', fontsize=13)\n",
        "ax.set_ylabel(f'Task Accuracy ({DATASET})', fontsize=13)\n",
        "ax.set_title(f'Task Accuracy vs. Knowledge Forgetting by Adapter Type\\n({DATASET} - {THRESHOLD})', fontsize=15, fontweight='bold')\n",
        "ax.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)\n",
        "ax.grid(True, alpha=0.3)\n",
        "\n",
        "# Annotation\n",
        "ax.annotate('Ideal: High accuracy,\\nlow forgetting',\n",
        "            xy=(subset['negative_shift_score'].min() * 1.1, subset['accuracy'].max()), \n",
        "            fontsize=10, fontstyle='italic',\n",
        "            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.3),\n",
        "            ha='center')\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()"
    ]
})

# =============================================================================
# CELL 7: Figure 2 - Size Comparison
# =============================================================================
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# FIGURE 2: Accuracy vs Negative Shift by Adapter SIZE\n",
        "fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)\n",
        "\n",
        "for idx, size in enumerate(['small', 'large']):\n",
        "    ax = axes[idx]\n",
        "    subset = df_merged[(df_merged['size'] == size) & (df_merged['negative_shift_score'] > 0)]\n",
        "    \n",
        "    for adapter in ['lora', 'vera', 'uiortholora']:\n",
        "        ad_sub = subset[subset['adapter'] == adapter]\n",
        "        if len(ad_sub) > 0:\n",
        "            ax.scatter(\n",
        "                ad_sub['negative_shift_score'],\n",
        "                ad_sub['accuracy'],\n",
        "                c=ADAPTER_COLORS.get(adapter, 'gray'),\n",
        "                label=ADAPTER_LABELS.get(adapter, adapter),\n",
        "                alpha=0.7,\n",
        "                s=90,\n",
        "                edgecolors='white',\n",
        "                linewidths=0.5,\n",
        "                marker=SIZE_MARKERS.get(size, 'o')\n",
        "            )\n",
        "    \n",
        "    ax.set_xlabel('Negative Shift Score', fontsize=12)\n",
        "    ax.set_title(f'{size.capitalize()} Adapters', fontsize=14, fontweight='bold')\n",
        "    ax.legend(fontsize=11, frameon=True)\n",
        "    ax.grid(True, alpha=0.3)\n",
        "\n",
        "axes[0].set_ylabel(f'Task Accuracy ({DATASET})', fontsize=12)\n",
        "fig.suptitle(f'Task Accuracy vs. Forgetting — Small vs. Large Adapters\\n({DATASET})', fontsize=15, fontweight='bold', y=1.02)\n",
        "plt.tight_layout()\n",
        "plt.show()"
    ]
})

# =============================================================================
# CELL 8: Figure 3 - LR Robustness
# =============================================================================
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# FIGURE 3: Learning Rate Robustness\n",
        "fig, ax = plt.subplots(figsize=(14, 9))\n",
        "\n",
        "lr_vals = sorted(df_merged[df_merged['negative_shift_score'] > 0]['lr'].dropna().unique())\n",
        "lr_cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(lr_vals)))\n",
        "lr_color_map = {lr: lr_cmap[i] for i, lr in enumerate(lr_vals)}\n",
        "\n",
        "marker_map = {'lora': 'o', 'vera': 's', 'uiortholora': '^'}\n",
        "\n",
        "for adapter in ['lora', 'vera', 'uiortholora']:\n",
        "    for lr_val in lr_vals:\n",
        "        subset = df_merged[(df_merged['adapter'] == adapter) & \n",
        "                           (df_merged['lr'] == lr_val) & \n",
        "                           (df_merged['negative_shift_score'] > 0)]\n",
        "        if len(subset) > 0:\n",
        "            ax.scatter(\n",
        "                subset['negative_shift_score'],\n",
        "                subset['accuracy'],\n",
        "                c=[lr_color_map[lr_val]],\n",
        "                marker=marker_map.get(adapter, 'o'),\n",
        "                s=100,\n",
        "                alpha=0.7,\n",
        "                edgecolors='gray',\n",
        "                linewidths=0.3,\n",
        "            )\n",
        "\n",
        "# Legends\n",
        "from matplotlib.lines import Line2D\n",
        "adapter_handles = [Line2D([0],[0], marker=marker_map.get(a, 'o'), color='gray', \n",
        "                          markersize=10, linestyle='None', label=ADAPTER_LABELS.get(a, a)) \n",
        "                   for a in ['lora','vera','uiortholora']]\n",
        "\n",
        "sm = plt.cm.ScalarMappable(cmap='viridis', norm=matplotlib.colors.LogNorm(vmin=1e-6, vmax=1e-2))\n",
        "sm.set_array([])\n",
        "cbar = fig.colorbar(sm, ax=ax, label='Learning Rate', shrink=0.8)\n",
        "\n",
        "ax.legend(handles=adapter_handles, fontsize=12, frameon=True, loc='lower right')\n",
        "ax.set_xlabel('Negative Shift Score', fontsize=13)\n",
        "ax.set_ylabel('Task Accuracy', fontsize=13)\n",
        "ax.set_title('Learning Rate Sensitivity', fontsize=15, fontweight='bold')\n",
        "ax.grid(True, alpha=0.3)\n",
        "plt.tight_layout()\n",
        "plt.show()"
    ]
})

# =============================================================================
# CELL 9: Figure 5 - Generalization (MMLU vs BigBench)
# =============================================================================
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# FIGURE 5: MMLU vs BigBench for well-performing models\n",
        "df_both = df_merged.dropna(subset=['bigbench_acc', 'mmlu_accuracy']).copy()\n",
        "\n",
        "if len(df_both) > 0:\n",
        "    fig, ax = plt.subplots(figsize=(12, 8))\n",
        "\n",
        "    for adapter in ['lora', 'vera', 'uiortholora']:\n",
        "        subset = df_both[df_both['adapter'] == adapter]\n",
        "        if len(subset) > 0:\n",
        "            ax.scatter(\n",
        "                subset['mmlu_accuracy'],\n",
        "                subset['bigbench_acc'],\n",
        "                c=ADAPTER_COLORS.get(adapter, 'gray'),\n",
        "                label=ADAPTER_LABELS.get(adapter, adapter),\n",
        "                alpha=0.7,\n",
        "                s=90,\n",
        "                edgecolors='white',\n",
        "                linewidths=0.5,\n",
        "            )\n",
        "\n",
        "    ax.set_xlabel('MMLU Accuracy', fontsize=13)\n",
        "    ax.set_ylabel('BigBench Mean Accuracy', fontsize=13)\n",
        "    ax.set_title(f'General Knowledge Retention: MMLU vs BigBench\\nafter {DATASET} Fine-tuning', fontsize=15, fontweight='bold')\n",
        "    ax.legend(fontsize=12, frameon=True)\n",
        "    ax.grid(True, alpha=0.3)\n",
        "\n",
        "    plt.tight_layout()\n",
        "    plt.show()\n",
        "else:\n",
        "    print(\"Not enough data with both BigBench and MMLU scores to plot Figure 5.\")"
    ]
})

# =============================================================================
# CELL 10: Figure 8 - Pareto Front (Generalization vs Forgetting)
# =============================================================================
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# FIGURE 8: BigBench vs Negative Shift\n",
        "df_pareto = df_merged.dropna(subset=['bigbench_acc']).copy()\n",
        "df_pareto = df_pareto[df_pareto['negative_shift_score'] > 0]\n",
        "\n",
        "if len(df_pareto) > 0:\n",
        "    fig, ax = plt.subplots(figsize=(13, 9))\n",
        "\n",
        "    for adapter in ['lora', 'vera', 'uiortholora']:\n",
        "        subset = df_pareto[df_pareto['adapter'] == adapter]\n",
        "        if len(subset) > 0:\n",
        "            ax.scatter(\n",
        "                subset['negative_shift_score'],\n",
        "                subset['bigbench_acc'],\n",
        "                c=ADAPTER_COLORS.get(adapter, 'gray'),\n",
        "                label=ADAPTER_LABELS.get(adapter, adapter),\n",
        "                alpha=0.7,\n",
        "                s=100,\n",
        "                edgecolors='white',\n",
        "                linewidths=0.5,\n",
        "            )\n",
        "\n",
        "    ax.set_xlabel('Negative Shift Score (Forgetting)', fontsize=13)\n",
        "    ax.set_ylabel('BigBench Mean Accuracy (General Knowledge)', fontsize=13)\n",
        "    ax.set_title(f'General Knowledge vs. Task-Specific Forgetting\\n({DATASET})', fontsize=15, fontweight='bold')\n",
        "    ax.legend(fontsize=12, frameon=True)\n",
        "    ax.grid(True, alpha=0.3)\n",
        "\n",
        "    plt.tight_layout()\n",
        "    plt.show()\n",
        "else:\n",
        "    print(\"Not enough data to plot BigBench vs Forgetting.\")"
    ]
})

# =============================================================================
# CELL 11: Summary Table
# =============================================================================
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# FIGURE 9: Summary table — best configs per adapter\n",
        "df_summary = df_merged.dropna(subset=['bigbench_acc', 'mmlu_accuracy']).copy()\n",
        "\n",
        "best_rows = []\n",
        "for adapter in ['lora', 'vera', 'uiortholora']:\n",
        "    sub = df_summary[df_summary['adapter'] == adapter]\n",
        "    if len(sub) == 0:\n",
        "        continue\n",
        "    \n",
        "    # Best by BigBench\n",
        "    best_bb = sub.loc[sub['bigbench_acc'].idxmax()]\n",
        "    best_rows.append({\n",
        "        'Adapter': ADAPTER_LABELS.get(adapter, adapter),\n",
        "        'LR': best_bb['lr'],\n",
        "        'Task Acc': f\"{best_bb['accuracy']:.3f}\",\n",
        "        'BigBench': f\"{best_bb['bigbench_acc']:.3f}\",\n",
        "        'MMLU': f\"{best_bb['mmlu_accuracy']:.3f}\",\n",
        "        'Neg Shift': f\"{best_bb['negative_shift_score']:.0f}\",\n",
        "        'Metric': 'Best BigBench'\n",
        "    })\n",
        "    \n",
        "    # Best by task acc with low forgetting (better than median)\n",
        "    median_shift = sub['negative_shift_score'].median()\n",
        "    low_forget = sub[sub['negative_shift_score'] < median_shift]\n",
        "    if len(low_forget) > 0:\n",
        "        best_task = low_forget.loc[low_forget['accuracy'].idxmax()]\n",
        "        best_rows.append({\n",
        "            'Adapter': ADAPTER_LABELS.get(adapter, adapter),\n",
        "            'LR': best_task['lr'],\n",
        "            'Task Acc': f\"{best_task['accuracy']:.3f}\",\n",
        "            'BigBench': f\"{best_task['bigbench_acc']:.3f}\",\n",
        "            'MMLU': f\"{best_task['mmlu_accuracy']:.3f}\",\n",
        "            'Neg Shift': f\"{best_task['negative_shift_score']:.0f}\",\n",
        "            'Metric': 'Best Task+Low Forget'\n",
        "        })\n",
        "\n",
        "df_best = pd.DataFrame(best_rows)\n",
        "print(\"SUMMARY: Best Configurations per Adapter\")\n",
        "df_best"
    ]
})

# =============================================================================
# SAVE NOTEBOOK
# =============================================================================
notebook_content = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.5"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

filename = 'adapter_analysis.ipynb'
with open(filename, 'w') as f:
    json.dump(notebook_content, f, indent=2)

print(f"Successfully created {filename}")