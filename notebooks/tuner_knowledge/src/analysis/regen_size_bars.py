#!/usr/bin/env python3
"""Regenerate the four camera-ready size-bar figures (INLG 2026 paper Figs 3, 7-9).

Rebuilt 2026-08-23 from comparison_analysis_sizes.ipynb after the original
camera-ready regen script was lost with its session scratchpad.

Selection logic is IDENTICAL to the published figures (verified value-by-value
against the 2026-08-21 PDFs before styling changes were applied):
  - model-restricted benchmark join (general rows filtered to the figure's model)
  - pool: runs with adaptation accuracy >= 0.98 that have benchmark evaluations
  - left panel: min negative_shift_score per method x tier (pool = task rows
    joined to MMLU, as in the notebook)
  - right panel: max mean(MMLU, BIG-bench-9) per method x tier

2026-08-23 design pass (values unchanged):
  - smaller in-figure title
  - External Retention axis starts near the data so differences are visible
  - thousands separators on Negative Shift labels; uniform tick formats
  - bottom tier legend removed (S/M/L row labels already encode the tier)
  - wider panel gap (no more tick-label collision between panels)

Run from notebooks/tuner_knowledge/src/analysis/:  python3 regen_size_bars.py OUTDIR
"""
import sys
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter

OUTDIR = sys.argv[1] if len(sys.argv) > 1 else './graphs_regen'

FS = 1.55  # camera-ready font scale (reviewer 2Fgf legibility fix)
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 9 * FS,
    'axes.labelsize': 10 * FS,
    'axes.titlesize': 10 * FS,
    'xtick.labelsize': 8 * FS,
    'ytick.labelsize': 8 * FS,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
})

SIZE_ORDER = ['Small', 'Medium', 'Large']
SIZE_COLORS = {'Small': '#66c2a5', 'Medium': '#fc8d62', 'Large': '#8da0cb'}
ADAPTER_DISPLAY = {'lora': 'LoRA', 'dora': 'DoRA', 'uiortholora': 'UIOrthoLoRA', 'vera': 'VeRA'}
NAME_RE = r'_(?:s|sv)(\d+)_(?:v|svec)(\d+)'


def extract_metadata(df, name_col='merge_key'):
    df = df.copy()

    def get_adapter(row):
        name = str(row[name_col]).lower()
        if 'uiortho' in name: return 'uiortholora'
        if 'dora' in name: return 'dora'
        if 'vera' in name: return 'vera'
        if 'lora' in name: return 'lora'
        return 'unknown'

    df['adapter_type'] = df.apply(get_adapter, axis=1)

    def get_svecs(row):
        m = re.search(NAME_RE, str(row[name_col]))
        return int(m.group(2)) if m else np.nan

    def get_rank(row):
        name = str(row[name_col])
        m_sv = re.search(NAME_RE, name)
        if m_sv: return int(m_sv.group(1))
        m_r = re.search(r'_r(\d+)', name)
        if m_r: return int(m_r.group(1))
        return 0

    df['svecs'] = df.apply(get_svecs, axis=1)
    df['rank'] = df.apply(get_rank, axis=1)
    return df


def assign_size_category(df):
    df = df.copy()

    def _get_size(row):
        atype = row['adapter_type']
        if atype == 'vera':
            return 'Small'
        elif atype in ('lora', 'dora'):
            r = row.get('rank', 0)
            if r <= 1: return 'Small'
            elif r <= 8: return 'Medium'
            else: return 'Large'
        elif atype == 'uiortholora':
            sv = row.get('svecs', 0)
            if pd.isna(sv): sv = 0
            if sv <= 32: return 'Small'
            elif sv < 512: return 'Medium'
            else: return 'Large'
        return 'Medium'

    df['size_category'] = df.apply(_get_size, axis=1)
    return df


HF_PREFIX = {'gemma-12b': 'google_gemma', 'llama-3b': 'meta-llama_Llama'}


def align_and_merge(df_task, df_general, model, general_name_col):
    """Notebook join + the camera-ready model restriction on the general side.

    Benchmark rows name the model either in the path (models/<model>/<task>/...)
    or in the adapter name itself (models/google_gemma-3-12b-it_...); keep both.
    """
    names = df_general[general_name_col]
    keep = names.str.contains(f'/{model}/', na=False) | \
        names.str.replace('models/', '', n=1).str.split('/').str[-1].str.startswith(HF_PREFIX[model])
    df_general = df_general[keep].copy()
    df_general['merge_key'] = (
        df_general[general_name_col].str.replace('models/', '', n=1).str.split('/').str[-1]
    )
    df_task = df_task.copy()
    df_task['merge_key'] = (
        df_task['raw_adapter_name'].str.replace('models/', '', n=1).str.split('/').str[-1]
    )
    merged = pd.merge(df_task, df_general, on='merge_key', suffixes=('_task', '_gen'))
    return extract_metadata(merged)


def load_and_merge(model, task):
    df_task = pd.read_csv(f'adapters_results/{task}/threshold_0.8/{model}.csv')
    df_mmlu = pd.read_csv(f'adapters_results/{task}/mmlu_summary.csv')
    df_bb = pd.read_csv(f'adapters_results/{task}/bigbench_summary.csv')
    merged_mmlu = align_and_merge(df_task, df_mmlu, model, 'model_name')
    merged_bb = align_and_merge(df_task, df_bb, model, 'model_path')
    return merged_mmlu, merged_bb


def get_best_per_size(merged_mmlu, merged_bb, acc_threshold=0.98):
    results = []
    src_neg = merged_mmlu
    adapters = sorted(set(merged_mmlu['adapter_type']) | set(merged_bb['adapter_type']))
    for adapter in adapters:
        neg_sub = src_neg[(src_neg['adapter_type'] == adapter) &
                          (src_neg['accuracy'] >= acc_threshold)].dropna(subset=['negative_shift_score'])
        neg_sub = assign_size_category(neg_sub)

        mmlu_sub = merged_mmlu[(merged_mmlu['adapter_type'] == adapter) &
                               (merged_mmlu['accuracy'] >= acc_threshold)].dropna(subset=['mmlu_acc'])
        mmlu_sub = assign_size_category(mmlu_sub)
        bb_sub = merged_bb[(merged_bb['adapter_type'] == adapter) &
                           (merged_bb['accuracy'] >= acc_threshold)].dropna(subset=['mean_accuracy'])
        bb_sub = assign_size_category(bb_sub)

        avg_parts = []
        if not mmlu_sub.empty:
            m = mmlu_sub[['merge_key', 'mmlu_acc', 'size_category']].rename(columns={'mmlu_acc': 'general_acc'})
            avg_parts.append(m)
        if not bb_sub.empty:
            b = bb_sub[['merge_key', 'mean_accuracy', 'size_category']].rename(columns={'mean_accuracy': 'general_acc'})
            avg_parts.append(b)
        if avg_parts:
            avg_df = (pd.concat(avg_parts, ignore_index=True)
                      .groupby('merge_key')
                      .agg(avg_general_acc=('general_acc', 'mean'),
                           size_category=('size_category', 'first'))
                      .reset_index())
        else:
            avg_df = pd.DataFrame()

        for size in SIZE_ORDER:
            row = {'adapter_type': adapter, 'size_category': size}
            ns = neg_sub[neg_sub['size_category'] == size]
            row['best_neg_shift'] = ns['negative_shift_score'].min() if not ns.empty else np.nan
            if not avg_df.empty:
                ag = avg_df[avg_df['size_category'] == size]
                row['best_avg_general'] = ag['avg_general_acc'].max() if not ag.empty else np.nan
            else:
                row['best_avg_general'] = np.nan
            if not (pd.isna(row['best_neg_shift']) and pd.isna(row['best_avg_general'])):
                results.append(row)
    return pd.DataFrame(results)


def plot_size_bars(best_df, model, task, save_path):
    model_label = model.replace('gemma-12b', 'Gemma-3-12B-IT').replace('llama-3b', 'Llama-3.2-3B-Instruct')
    task_label = task.replace('hotpotqa', 'HotpotQA').replace('triviaqa', 'TriviaQA')

    all_adapters = sorted(best_df['adapter_type'].unique())
    y_positions, y = [], 0
    bar_height = 0.65
    adapter_centers = {}
    for adapter in all_adapters:
        adf = best_df[best_df['adapter_type'] == adapter]
        sizes_present = [s for s in SIZE_ORDER if s in adf['size_category'].values]
        start_y = y
        for size in sizes_present:
            y_positions.append((y, adapter, size))
            y += 1
        adapter_centers[adapter] = start_y
        y += 0.8

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 3.8), gridspec_kw={'wspace': 0.34})
    fig.suptitle(f'{model_label} | {task_label}  (adaptation $\\geq$ 0.98)',
                 fontsize=8.5 * FS, fontweight='bold')

    metrics = [('best_neg_shift', 'Negative Shift'),
               ('best_avg_general', 'External Retention ($R_{\\mathrm{ext}}$)')]

    for col_i, (metric, ptitle) in enumerate(metrics):
        ax = axes[col_i]
        ax.grid(True, axis='x', linestyle='-', linewidth=0.3, alpha=0.35, color='#bbbbbb')

        vals = []
        for yp, adapter, size in y_positions:
            row = best_df[(best_df['adapter_type'] == adapter) &
                          (best_df['size_category'] == size)]
            if row.empty or pd.isna(row[metric].values[0]):
                continue
            vals.append(row[metric].values[0])

        if metric == 'best_neg_shift':
            lo, hi = 0.0, max(vals) * 1.02
            span = hi - lo
        else:
            lo = max(0.0, np.floor((min(vals) - 0.03) * 20) / 20)
            hi = max(vals) + 0.005
            span = hi - lo

        for yp, adapter, size in y_positions:
            row = best_df[(best_df['adapter_type'] == adapter) &
                          (best_df['size_category'] == size)]
            if row.empty or pd.isna(row[metric].values[0]):
                continue
            val = row[metric].values[0]
            ax.barh(yp, val - lo, left=lo, height=bar_height,
                    color=SIZE_COLORS[size], edgecolor='#444444',
                    linewidth=0.4, zorder=3)
            fmt = f'{val:,.0f}' if metric == 'best_neg_shift' else f'{val:.3f}'
            ax.text(val + 0.015 * span, yp, fmt, va='center', ha='left',
                    fontsize=6.5 * FS, color='#333')

        all_y = [yp for yp, _, _ in y_positions]
        ax.set_yticks(all_y)
        if col_i == 0:
            ax.set_yticklabels([size[0] for _, _, size in y_positions], fontsize=7.5 * FS)
        else:
            ax.set_yticklabels([])
        ax.invert_yaxis()
        ax.set_title(ptitle, fontsize=8.5 * FS, pad=4)

        ax.set_xlim(lo, hi + 0.22 * span)
        if metric == 'best_neg_shift':
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 5, 10]))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:,.0f}'))
        else:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:.2f}'))

        for ai in range(len(all_adapters) - 1):
            adapter, next_adapter = all_adapters[ai], all_adapters[ai + 1]
            last_y = max(yp for yp, a, _ in y_positions if a == adapter)
            first_next_y = min(yp for yp, a, _ in y_positions if a == next_adapter)
            ax.axhline((last_y + first_next_y) / 2, color='#cccccc', linewidth=0.5)

    # Method name as a horizontal bold header above each group (left panel);
    # rotated right-side labels overflowed for the longest name (UIOrthoLoRA).
    for ax in axes:
        lo_lim, hi_lim = ax.get_ylim()
        ax.set_ylim(lo_lim, -1.05)
    for adapter, start_y in adapter_centers.items():
        axes[0].annotate(ADAPTER_DISPLAY.get(adapter, adapter), xy=(0.0, start_y - 0.68),
                         xycoords=('axes fraction', 'data'),
                         fontsize=6.8 * FS, fontweight='bold',
                         ha='left', va='center', color='#222222')

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f'Saved: {save_path}')


# Published bar values (read off the 2026-08-21 camera-ready PDFs).
# The regeneration must reproduce these exactly; any mismatch aborts.
EXPECTED = {
    ('gemma-12b', 'hotpotqa'): {
        ('dora', 'Small'): (1571, 0.661), ('dora', 'Medium'): (1461, 0.671), ('dora', 'Large'): (1514, 0.664),
        ('lora', 'Small'): (1589, 0.654), ('lora', 'Medium'): (1468, 0.664), ('lora', 'Large'): (1497, 0.669),
        ('uiortholora', 'Small'): (1548, 0.653), ('uiortholora', 'Medium'): (1492, 0.675), ('uiortholora', 'Large'): (1523, 0.654),
        ('vera', 'Small'): (1624, 0.634),
    },
    ('gemma-12b', 'triviaqa'): {
        ('dora', 'Small'): (8923, 0.664), ('dora', 'Medium'): (8114, 0.667), ('dora', 'Large'): (7604, 0.674),
        ('lora', 'Small'): (8920, 0.651), ('lora', 'Medium'): (7825, 0.672), ('lora', 'Large'): (7639, 0.678),
        ('uiortholora', 'Small'): (8523, 0.646), ('uiortholora', 'Medium'): (8430, 0.653), ('uiortholora', 'Large'): (8282, 0.657),
        ('vera', 'Small'): (10022, 0.641),
    },
    ('llama-3b', 'hotpotqa'): {
        ('dora', 'Medium'): (982, 0.511), ('dora', 'Large'): (926, 0.518),
        ('lora', 'Medium'): (972, 0.504), ('lora', 'Large'): (872, 0.522),
        ('uiortholora', 'Small'): (1415, 0.295), ('uiortholora', 'Medium'): (1276, 0.344), ('uiortholora', 'Large'): (1215, 0.331),
    },
    ('llama-3b', 'triviaqa'): {
        ('dora', 'Medium'): (14582, 0.450), ('dora', 'Large'): (14289, 0.476),
        ('lora', 'Medium'): (9424, 0.534), ('lora', 'Large'): (8783, 0.531),
        ('uiortholora', 'Medium'): (15678, 0.350), ('uiortholora', 'Large'): (15247, 0.359),
    },
}


def verify(best_df, model, task):
    exp = EXPECTED[(model, task)]
    got = {}
    for _, r in best_df.iterrows():
        if pd.isna(r['best_neg_shift']) and pd.isna(r['best_avg_general']):
            continue
        got[(r['adapter_type'], r['size_category'])] = (r['best_neg_shift'], r['best_avg_general'])
    errors = []
    if set(got) != set(exp):
        errors.append(f'bar set differs: extra={set(got)-set(exp)}, missing={set(exp)-set(got)}')
    for key in sorted(set(got) & set(exp)):
        ns_g, r_g = got[key]
        ns_e, r_e = exp[key]
        if round(ns_g) != ns_e:
            errors.append(f'{key}: NS {ns_g:.1f} (rounds {round(ns_g)}) != published {ns_e}')
        if f'{r_g:.3f}' != f'{r_e:.3f}':
            errors.append(f'{key}: R_ext {r_g:.4f} != published {r_e:.3f}')
    return errors


def main():
    import os
    os.makedirs(OUTDIR, exist_ok=True)
    all_ok = True
    for model in ['gemma-12b', 'llama-3b']:
        for task in ['hotpotqa', 'triviaqa']:
            merged_mmlu, merged_bb = load_and_merge(model, task)
            best_df = get_best_per_size(merged_mmlu, merged_bb, acc_threshold=0.98)
            errs = verify(best_df, model, task)
            if errs:
                all_ok = False
                print(f'!! {model}/{task}: {len(errs)} MISMATCHES vs published figures:')
                for e in errs:
                    print('   ', e)
            else:
                print(f'OK {model}/{task}: all bars match the published figures exactly')
                plot_size_bars(best_df, model, task, f'{OUTDIR}/{model}_{task}_size_bars.pdf')
    if not all_ok:
        sys.exit(1)


if __name__ == '__main__':
    main()
