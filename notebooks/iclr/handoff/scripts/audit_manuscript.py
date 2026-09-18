"""Read-only manuscript/data/build audit. Does not certify unarchived training runs."""
import argparse
from collections import Counter
import csv
import hashlib
from decimal import Decimal
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
    parser.add_argument('--review', action='store_true',
                        help='Allow abstract, introduction and RTE study review markup and colored PDF.')
    args = parser.parse_args()
    path, expected = manuscript_output()
    raw = path.read_text()
    assert raw == expected, 'Generated regions differ from source data.'
    text = uncomment(raw)
    assert not re.search(r'\\(?:input|include|tablerows|includegraphics)\s*\{', text)
    markup = r'\\(?:new|cut)\s*\{|\\begin\{newpart\}|\\difftrue\b'
    assert not re.search(r'\\iclrfinalcopy\b', text)
    if args.review:
        definitions = re.search(r'% BEGIN INTRODUCTION REVIEW DEFINITIONS\n.*?% END INTRODUCTION REVIEW DEFINITIONS\n', raw, re.S)
        assert definitions, 'Missing inline review definitions.'
        body = uncomment(raw[:definitions.start()] + raw[definitions.end():])
        intro_start = body.index(r'\section{Introduction}')
        intro_end = body.index(r'\section{', intro_start + 1)
        outside_intro = body[:intro_start] + body[intro_end:]
        outside_review = re.sub(r'\\begin\{abstract\}.*?\\end\{abstract\}', '',
                                outside_intro, count=1, flags=re.S)
        for label in ('sec:tailstudy', 'app:bandstudy'):
            start = outside_review.index(r'\label{'+label+'}')
            end = outside_review.index(r'\subsection{', start)
            outside_review = outside_review[:start] + outside_review[end:]
        assert not re.search(markup, outside_review), 'Review markup outside abstract, introduction and RTE study.'
    else:
        assert not re.search(markup, text), 'Pending review markup: use --review for an author-review audit.'
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
    mode = 'abstract, introduction and RTE study review markup allowed' if args.review else 'without review markup'
    print(f'PASS: sole manuscript source; {len(labels)} unique labels; {len(citations)} cited keys; balanced environments/braces; anonymous, {mode}.')

    # Preserve supporting numeric table bodies through the consolidation.
    def old(name):
        fixture = ROOT / 'audit_baseline' / name
        if fixture.exists():
            expected = {
                'tables/generation.tex': 'b570e98f196b19c3788e071c81d6ee126209d85d5ac89cea5f0a8a6808922181',
                'forgetting-results-table.tex': '2abd180dca6f0ab2a0c76b90a1827610498c3b42f54d0cbac600d7a27e9c8a72',
                'tables/glue.tex': 'b072a625b00fefca35392545b4c5f045199c3971fde1571d1244aebca1f0d816',
            }
            assert hashlib.sha256(fixture.read_bytes()).hexdigest() == expected[name]
            return fixture.read_text()
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
    print('PASS: all 60 GLUE means and 60 standard deviations preserved; one MRPC accuracy column; six-task averages audited in source only.')

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
        end_pages = [i+1 for i,p in enumerate(pdf) if p.search_for('Controlling interaction changes how the model adapts')]
        # Review copies include superseded paragraphs; clean copies retain the page limit.
        has_blue = any(span['color'] >> 16 != (span['color'] >> 8)&255
                       for page in pdf for block in page.get_text('dict')['blocks']
                       for line in block.get('lines', []) for span in line['spans'])
        review_pdf = args.review and has_blue
        if review_pdf:
            assert len(end_pages) == 1, ('Missing/duplicate main conclusion', end_pages)
        else:
            assert end_pages == [9], ('Main conclusion must finish on page 9',end_pages)
        glue_pages = [i+1 for i,p in enumerate(pages) if 'RandLoRA-100' in p]
        assert glue_pages and max(glue_pages) <= end_pages[0]
        for page in pdf:
            for block in page.get_text('dict')['blocks']:
                for line in block.get('lines', []):
                    for span in line['spans']:
                        color = span['color']
                        rgb = (color >> 16, (color >> 8)&255, color&255)
                        gray = rgb[0] == rgb[1] == rgb[2]
                        blue = all(abs(a-b) <= 1 for a,b in zip(rgb, (0,65,170)))
                        assert gray or (review_pdf and blue), color
            assert not re.search(r'olp_[A-Za-z0-9]+|6aa54397e58b10444b0fa2aa|/home/', page.get_text())
        colors = 'blue/gray author review' if review_pdf else 'grayscale text'
        print(f'PASS: PDF {len(pdf)} pages; main conclusion on page {end_pages[0]}; full GLUE on page {glue_pages[0]}; no undefined/overflow/missing-glyph errors; {colors}.')
        print('NOTE: underfull spacing and Tectonic bibliography-rerun notices are not counted as missing-reference or overflow errors.')
    print('LIMIT: no training rerun; broad GLUE seed/config/count provenance and causal/modern-model controls remain open. See MANUSCRIPT_AUDIT.md.')


if __name__ == '__main__':
    main()
