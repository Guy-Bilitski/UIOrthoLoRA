"""Read-only manuscript/data/build audit. Does not certify unarchived training runs."""
import argparse
from collections import Counter
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
from statistics import mean
import subprocess

from analyze_archived_layers import analyze, records, key
from build_mixing_tables import manuscript_output

ROOT = Path(__file__).resolve().parents[1]
BASELINE = '9c8043c1b531f3afb27fa55ce101c8635736c938'


def uncomment(text):
    return re.sub(r'(?<!\\)%[^\n]*', '', text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pdf', type=Path)
    args = parser.parse_args()
    path, expected = manuscript_output()
    raw = path.read_text()
    assert raw == expected, 'Generated regions differ from source data.'
    text = uncomment(raw)
    assert not re.search(r'\\(?:input|include|tablerows|includegraphics)\s*\{', text)
    assert not re.search(r'\\(?:new|cut)\s*\{|\\begin\{newpart\}|\\(?:difftrue|iclrfinalcopy)\b', text)
    assert r'\author{Anonymous authors}' in text
    assert not re.search(r'olp_[A-Za-z0-9]+|/home/|6aa54397e58b10444b0fa2aa', text)
    labels = re.findall(r'\\label\{([^}]+)\}', text)
    assert not [label for label, n in Counter(labels).items() if n > 1]
    refs = set(re.findall(r'\\(?:eqref|ref|pageref)\{([^}]+)\}', text))
    assert refs <= set(labels), refs-set(labels)
    citations = {k.strip() for c in re.findall(r'\\cite\w*\{([^}]+)\}', text) for k in c.split(',')}
    bib_keys = re.findall(r'@\w+\{\s*([^,]+),', (ROOT / 'reference.bib').read_text())
    assert citations <= set(bib_keys), citations-set(bib_keys)
    assert not [k for k, n in Counter(bib_keys).items() if n > 1]
    stack = []
    for match in re.finditer(r'\\(begin|end)\{([^}]+)\}', text):
        kind, env = match.groups()
        if kind == 'begin':
            stack.append(env)
        else:
            assert stack and stack.pop() == env, (match.start(), env)
    assert not stack
    depth = 0
    for char in re.sub(r'\\[{}]', '', text):
        depth += (char == '{') - (char == '}')
        assert depth >= 0
    assert depth == 0
    print(f'PASS: sole manuscript source; {len(labels)} unique labels; {len(citations)} cited keys; balanced environments/braces; anonymous, without review markup.')

    # Preserve supporting numeric table bodies through the consolidation.
    def old(name):
        fixtures = ROOT / 'audit_baseline'
        manifest_path = fixtures / 'manifest.json'
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            assert manifest['overleaf_commit'] == BASELINE, 'Unexpected audit baseline.'
            assert name in manifest['sha256'], f'Unrecorded audit fixture: {name}'
            data = (fixtures / name).read_bytes()
            assert hashlib.sha256(data).hexdigest() == manifest['sha256'][name], f'Changed audit fixture: {name}'
            return data.decode('utf-8')
        return subprocess.check_output(['git', 'show', f'{BASELINE}:{name}'], cwd=ROOT, text=True)
    for label, filename in [('tab:gpt2_generation_appendix', 'tables/generation.tex'),
                            ('tab:combined_model_performance', 'forgetting-results-table.tex')]:
        source = raw if label == 'tab:gpt2_generation_appendix' else (ROOT / 'archive/retention_2026_09_14.tex').read_text()
        current = next(f for f in re.findall(r'\\begin\{table\*?\}.*?\\end\{table\*?\}', source, re.S)
                       if r'\label{' + label + '}' in f)
        def numbers(source):
            bodies = re.findall(r'\\begin\{tabular\}.*?\\end\{tabular\}', source, re.S)
            return re.findall(r'\d+(?:\.\d+)?', ''.join(bodies))
        assert numbers(current) == numbers(old(filename)), label
    assert 'tab:combined_model_performance' not in raw
    print('PASS: generation numeric table preserved; prior retention table preserved in inactive archive, absent from manuscript.')

    with (ROOT / 'data/glue_reported.csv').open(newline='') as handle:
        glue = list(csv.DictReader(handle))
    old_glue = old('tables/glue.tex')
    prior_rows = [line for line in old_glue.splitlines()
                  if re.match(r'^(VeRA|LoRA|RandLoRA|UIOrthoLoRA|UILinLoRA)', line)]
    assert len(prior_rows) == len(glue) == 10
    for row, previous in zip(glue, prior_rows):
        cells = [c.strip() for c in previous.split('&')]
        assert cells[0] == row['method'] and cells[1] == row['reported_params']
        for task, cell in zip(('sst2', 'mrpc', 'cola', 'qnli', 'rte', 'stsb'), cells[2:8]):
            values = re.findall(r'\d+(?:\.\d+)?', cell)
            assert values == [row[task+'_mean'], row[task+'_sd']], (row['method'], task)
        assert row['mrpc_metric'] == 'accuracy'
        avg = sum(Decimal(row[t+'_mean']) for t in ('sst2','mrpc','cola','qnli','rte','stsb'))/6
        print(f'GLUE {row["backbone"]} {row["method"]}: Avg6={avg:.2f}, MRPC={row["mrpc_metric"]}')
    glue_table = next(f for f in re.findall(r'\\begin\{table\*?\}.*?\\end\{table\*?\}', text, re.S)
                      if r'\label{tab:glue_results_base_large}' in f)
    assert 'MRPC Acc. & CoLA' in glue_table and 'MRPC F1' not in glue_table
    assert 'Avg5' not in text
    print('PASS: all 60 GLUE means and 60 standard deviations preserved; one MRPC accuracy column and six-task averages.')

    values, report = analyze()
    assert report['cross_reduced_runs'] == 9
    assert all(n == 9 for n in report['run_mean_reductions'].values())
    assert report['module_reductions'] == {
        'pLL': 119, 'pCross': 410, 'OffTailRatio_F': 332,
        'RelPert_F': 429, 'Drift_U': 260, 'Drift_V': 256}
    tasks = {t for t, _ in values}
    assert sum(values[t,'C_muE_nuD']['pLL'] > values[t,'A_no_reg']['pLL'] for t in tasks) == 4
    summaries = {key(r): r for r in records('legacy_mixing')}
    score_changes = []
    task_score_changes = []
    for task in tasks:
        seed_changes = []
        for seed in (42,) if task == 'sst2_lin' else (17,42):
            a,b,c = [summaries[task, condition, seed] for condition in ('A_no_reg','B_muE_only','C_muE_nuD')]
            assert float(b['mean_mu_E']) < float(a['mean_mu_E'])
            assert float(b['mean_nu_D']) > float(a['mean_nu_D'])
            score = lambda r: float(r.get('final_val_score', r.get('final_val_accuracy')))
            seed_changes.append(100*(score(c)-score(a)))
        score_changes.extend(seed_changes)
        task_score_changes.append(mean(seed_changes))
    assert (sum(x<0 for x in score_changes),sum(x==0 for x in score_changes),sum(x>0 for x in score_changes)) == (7,1,1)
    print(f'PASS: all stated paired/module counts; task-mean score changes {min(task_score_changes):.4f} to {max(task_score_changes):.4f} points.')
    print(f'PASS: layer-to-summary reconciliation, maximum absolute error {report["max_summary_absolute_error"]:.3g}.')

    if args.pdf:
        import fitz
        pdf = fitz.open(args.pdf)
        pages = [page.get_text() for page in pdf]
        log = args.pdf.with_suffix('.log').read_text()
        assert not re.search(r'Overfull \\[hv]box|undefined|Missing character|LaTeX Error|^!', log, re.M|re.I)
        assert all('??' not in page for page in pages)
        end_pages = [i+1 for i,p in enumerate(pdf) if p.search_for('which adapter scores highest')]
        assert end_pages == [9], ('Main conclusion must finish on page 9',end_pages)
        glue_pages = [i+1 for i,p in enumerate(pages) if 'RandLoRA-100' in p]
        assert glue_pages and max(glue_pages) <= 9
        for page in pdf:
            for block in page.get_text('dict')['blocks']:
                for line in block.get('lines', []):
                    for span in line['spans']:
                        color = span['color']
                        assert color >> 16 == (color >> 8)&255 == color&255, color
            assert not re.search(r'olp_[A-Za-z0-9]+|6aa54397e58b10444b0fa2aa|/home/', page.get_text())
        print(f'PASS: PDF {len(pdf)} pages; main conclusion on page 9; full GLUE on page {glue_pages[0]}; no undefined/overflow/missing-glyph errors; grayscale text.')
        print('NOTE: underfull spacing and Tectonic bibliography-rerun notices are not counted as missing-reference or overflow errors.')
    print('LIMIT: no training rerun; broad GLUE seed/config/count provenance and causal/modern-model controls remain open. See MANUSCRIPT_AUDIT.md.')


if __name__ == '__main__':
    main()
