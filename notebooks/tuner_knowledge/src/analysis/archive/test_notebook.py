#!/usr/bin/env python
"""
Test script to validate the extrinsic analysis notebook.
This runs the key analysis steps comparing adapters across 3 models.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import re
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 100)
print("TESTING EXTRINSIC ANALYSIS NOTEBOOK - ADAPTER COMPARISON BY MODEL")
print("=" * 100)

# Load data
DATA_PATH = Path('../results/mmlu/mmlu_summary.csv')
print(f"\n1. Loading data from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print(f"   ✓ Loaded {len(df)} rows")

# Parse model names
def parse_model_name(model_name):
    result = {}

    if 'Llama-3.2-3B' in model_name:
        result['model'] = 'llama-3b'
        result['model_full'] = 'Llama-3.2-3B'
    elif 'Llama-3.1-8B' in model_name:
        result['model'] = 'llama-8b'
        result['model_full'] = 'Llama-3.1-8B'
    elif 'gemma-3-12b' in model_name or 'gemma-2-12b' in model_name:
        result['model'] = 'gemma-12b'
        result['model_full'] = 'Gemma-3-12B'
    else:
        result['model'] = 'unknown'
        result['model_full'] = 'Unknown'

    adapter_pattern = r'_(lora|vera|uiortholora|randlora)_tr'
    adapter_match = re.search(adapter_pattern, model_name)
    if adapter_match:
        result['adapter'] = adapter_match.group(1)
    else:
        result['adapter'] = 'unknown'

    tr_pattern = r'_tr(\d+)_'
    tr_match = re.search(tr_pattern, model_name)
    if tr_match:
        result['tr'] = int(tr_match.group(1))
    else:
        result['tr'] = None

    lr_pattern = r'_lr(\d+e-\d+)'
    lr_match = re.search(lr_pattern, model_name)
    if lr_match:
        result['lr'] = lr_match.group(1)
    else:
        result['lr'] = 'unknown'

    return result

print("\n2. Parsing model names...")
parsed_info = df['model_name'].apply(parse_model_name)
df_parsed = pd.DataFrame(parsed_info.tolist())
df_combined = pd.concat([df, df_parsed], axis=1)
print(f"   ✓ Parsed models: {sorted(df_combined['model'].unique())}")
print(f"   ✓ Parsed adapters: {sorted(df_combined['adapter'].unique())}")
print(f"   ✓ Parsed learning rates: {sorted(df_combined['lr'].unique())}")

# Main analysis
print("\n3. Computing average accuracy by model, adapter, and learning rate...")
avg_acc_by_model_adapter_lr = df_combined.groupby(['model', 'model_full', 'adapter', 'lr']).agg({
    'mmlu_acc': ['mean', 'std', 'count'],
    'mmlu_humanities_acc': 'mean',
    'mmlu_other_acc': 'mean',
    'mmlu_social_sciences_acc': 'mean',
    'mmlu_stem_acc': 'mean'
}).round(4)

avg_acc_by_model_adapter_lr.columns = ['_'.join(col).strip() if col[1] else col[0]
                                        for col in avg_acc_by_model_adapter_lr.columns.values]
avg_acc_by_model_adapter_lr = avg_acc_by_model_adapter_lr.reset_index()
avg_acc_by_model_adapter_lr.rename(columns={
    'mmlu_acc_mean': 'avg_mmlu_acc',
    'mmlu_acc_std': 'std_mmlu_acc',
    'mmlu_acc_count': 'num_experiments'
}, inplace=True)

print(f"   ✓ Computed {len(avg_acc_by_model_adapter_lr)} model-adapter-lr combinations")

# Show top results per model
models = sorted(avg_acc_by_model_adapter_lr['model'].unique())
model_full_names = {}
for model in models:
    model_full_names[model] = avg_acc_by_model_adapter_lr[avg_acc_by_model_adapter_lr['model'] == model]['model_full'].iloc[0]

print("\n4. Top configurations per model:")
print("   " + "-" * 96)
for model in models:
    model_data = avg_acc_by_model_adapter_lr[avg_acc_by_model_adapter_lr['model'] == model]
    top_3 = model_data.nlargest(3, 'avg_mmlu_acc')
    print(f"\n   {model_full_names[model]}:")
    for idx, row in top_3.iterrows():
        print(f"      {row['adapter']:15s} | lr={row['lr']:8s} | "
              f"avg_acc={row['avg_mmlu_acc']:.4f} ± {row['std_mmlu_acc']:.4f} "
              f"(n={int(row['num_experiments'])})")

# Create test visualizations
print("\n5. Creating visualizations...")
graphs_dir = Path('graphs')
graphs_dir.mkdir(exist_ok=True)

# Combined 3-model comparison
fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)

for idx, model in enumerate(models):
    ax = axes[idx]
    model_data = avg_acc_by_model_adapter_lr[avg_acc_by_model_adapter_lr['model'] == model]

    adapters = sorted(model_data['adapter'].unique())
    learning_rates = sorted(model_data['lr'].unique())

    x = np.arange(len(adapters))
    width = 0.25

    for i, lr in enumerate(learning_rates):
        lr_data = model_data[model_data['lr'] == lr]
        lr_data = lr_data.set_index('adapter').reindex(adapters)

        bars = ax.bar(x + i*width, lr_data['avg_mmlu_acc'], width,
                     label=f'lr={lr}', alpha=0.8,
                     yerr=lr_data['std_mmlu_acc'], capsize=4)

        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Adapter Type', fontsize=11, fontweight='bold')
    ax.set_title(f'{model_full_names[model]}', fontsize=13, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(adapters, rotation=0)
    ax.legend(title='Learning Rate', fontsize=9, loc='lower right')
    ax.grid(axis='y', alpha=0.3)

axes[0].set_ylabel('Average MMLU Accuracy\\n(across all training samples)', fontsize=11, fontweight='bold')
fig.suptitle('Adapter Performance Comparison Across Models and Learning Rates',
             fontsize=15, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('graphs/adapter_comparison_by_model_lr.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Created combined graph: graphs/adapter_comparison_by_model_lr.png")

# Individual model graphs
for model in models:
    fig, ax = plt.subplots(figsize=(10, 7))
    model_data = avg_acc_by_model_adapter_lr[avg_acc_by_model_adapter_lr['model'] == model]

    adapters = sorted(model_data['adapter'].unique())
    learning_rates = sorted(model_data['lr'].unique())

    x = np.arange(len(adapters))
    width = 0.25

    for i, lr in enumerate(learning_rates):
        lr_data = model_data[model_data['lr'] == lr]
        lr_data = lr_data.set_index('adapter').reindex(adapters)

        ax.bar(x + i*width, lr_data['avg_mmlu_acc'], width,
               label=f'lr={lr}', alpha=0.8,
               yerr=lr_data['std_mmlu_acc'], capsize=5)

    ax.set_xlabel('Adapter Type', fontsize=13, fontweight='bold')
    ax.set_ylabel('Average MMLU Accuracy\\n(across all training samples)', fontsize=13, fontweight='bold')
    ax.set_title(f'{model_full_names[model]} - Adapter Performance by Learning Rate',
                fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(adapters, rotation=0, fontsize=11)
    ax.legend(title='Learning Rate', fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    filename = f'graphs/{model}_adapter_comparison_by_lr.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Created {model} graph: {filename}")

# Alternative view: grouped by learning rate
print("\n   Creating alternative view (grouped by learning rate)...")
fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)

for idx, model in enumerate(models):
    ax = axes[idx]
    model_data = avg_acc_by_model_adapter_lr[avg_acc_by_model_adapter_lr['model'] == model]

    adapters = sorted(model_data['adapter'].unique())
    learning_rates = sorted(model_data['lr'].unique())

    x = np.arange(len(learning_rates))
    width = 0.20

    for i, adapter in enumerate(adapters):
        adapter_data = model_data[model_data['adapter'] == adapter]
        adapter_data = adapter_data.set_index('lr').reindex(learning_rates)

        bars = ax.bar(x + i*width, adapter_data['avg_mmlu_acc'], width,
                     label=adapter, alpha=0.8,
                     yerr=adapter_data['std_mmlu_acc'], capsize=4)

        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Learning Rate', fontsize=11, fontweight='bold')
    ax.set_title(f'{model_full_names[model]}', fontsize=13, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(learning_rates, rotation=0)
    ax.legend(title='Adapter', fontsize=9, loc='lower right')
    ax.grid(axis='y', alpha=0.3)

axes[0].set_ylabel('Average MMLU Accuracy\\n(across all training samples)', fontsize=11, fontweight='bold')
fig.suptitle('Adapter Comparison Grouped by Learning Rate',
             fontsize=15, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('graphs/adapter_comparison_grouped_by_lr.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Created alternative view: graphs/adapter_comparison_grouped_by_lr.png")

# Export results
print("\n6. Exporting summary results...")
output_path = Path('results_summary')
output_path.mkdir(exist_ok=True)

avg_acc_by_model_adapter_lr.to_csv(output_path / 'avg_accuracy_by_model_adapter_lr.csv', index=False)
print(f"   ✓ Saved: {output_path / 'avg_accuracy_by_model_adapter_lr.csv'}")

# Summary statistics
print("\n" + "=" * 100)
print("SUMMARY STATISTICS")
print("=" * 100)

print("\nBest configuration per model:")
print("-" * 100)
for model in models:
    model_avg_data = avg_acc_by_model_adapter_lr[avg_acc_by_model_adapter_lr['model'] == model]
    best = model_avg_data.loc[model_avg_data['avg_mmlu_acc'].idxmax()]
    print(f"{model_full_names[model]:20s} | {best['adapter']:15s} | lr={best['lr']:8s} | acc={best['avg_mmlu_acc']:.4f} ± {best['std_mmlu_acc']:.4f}")

print("\nOverall model performance (averaged across all adapters, lr, and tr):")
print("-" * 100)
model_overall = df_combined.groupby('model')['mmlu_acc'].agg(['mean', 'std', 'max', 'count']).round(4)
model_overall = model_overall.sort_values('mean', ascending=False)
for model, row in model_overall.iterrows():
    print(f"{model_full_names[model]:20s} | mean={row['mean']:.4f} | std={row['std']:.4f} | max={row['max']:.4f} | n={int(row['count'])}")

print("\n" + "=" * 100)
print("ALL TESTS PASSED! ✓")
print("=" * 100)
print("\nYou can now run the full notebook 'extrinsic_analysis.ipynb'")
print("to see all visualizations and detailed analysis for each of the 3 models.")
