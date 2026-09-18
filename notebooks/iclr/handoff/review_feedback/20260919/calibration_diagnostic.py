"""Descriptive calibration from archived binary, unsmoothed per-example CE.

No temperature fitting, checkpoint selection, inference, or training.
The executed evaluator uses F.cross_entropy(logits, labels, reduction='none').
For two classes, p(y|x)=exp(-loss) determines both class probabilities.
These reconstructed probabilities retain the precision limits of float32 CE.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper-root', required=True, type=Path,
                        help='Current Overleaf root or synchronized GitHub handoff root')
    parser.add_argument('--out', required=True, type=Path,
                        help='New result JSON; do not overwrite the archived diagnostic')
    args = parser.parse_args()
    ROOT = args.paper_root.resolve()
    DATA = ROOT / 'data/locked_evaluation_20260916'
    if args.out.exists():
        parser.error('Output exists; choose a fresh path to preserve recorded results.')
    manifest = json.loads((DATA / 'evaluation_manifest.json').read_text())
    evaluator = (HERE / 'locked_evaluation_source.txt').read_bytes()
    assert hashlib.sha256(evaluator).hexdigest() == manifest['script_sha256']
    assert b'cross_entropy(logits, batch["labels"], reduction="none")' in evaluator
    assert sha(DATA / 'held_aside_results.csv') == manifest['results_csv_sha256']
    records = {r['run_id']: r for r in csv.DictReader((DATA / 'held_aside_results.csv').open())}
    per_run, paired_splits = [], {}
    for path in sorted((DATA / 'per_example').glob('*.json')):
        d = json.loads(path.read_text())
        r = records[d['run_id']]
        assert d['checkpoint_sha256'] == r['checkpoint_sha256']
        assert d['split_fingerprint'] == r['locked_split_fingerprint']
        y, pred, losses = d['labels'], d['predictions'], d['per_example_loss']
        n = len(y)
        assert n == int(r['example_count']) == (277 if d['task'] == 'rte' else 408)
        assert len(pred) == len(losses) == len(set(d['sample_ids'])) == n
        assert set(y) == {0, 1} and set(pred) <= {0, 1}
        assert all(math.isfinite(v) and v >= 0 for v in losses)
        split = (d['sample_ids'], y)
        assert paired_splits.setdefault(d['task'], split) == split
        correct = [a == b for a, b in zip(y, pred)]
        accuracy, nll = statistics.mean(correct), statistics.mean(losses)
        assert abs(accuracy - float(r['held_aside_accuracy'])) < 1e-12
        assert abs(nll - float(r['held_aside_loss'])) < 1e-6
        ptrue = [math.exp(-v) for v in losses]
        assert all((p >= .5) == c for p, c in zip(ptrue, correct))
        confidence = [max(p, 1-p) for p in ptrue]
        # Equal-width bins (left, right], matching the existing temperature analyzer.
        bins = [[] for _ in range(15)]
        for p, c in zip(confidence, correct):
            bins[max(0, min(14, math.ceil(15 * p)-1))].append((p, c))
        ece = sum(len(b)/n * abs(statistics.mean(x[0] for x in b)
                               - statistics.mean(x[1] for x in b)) for b in bins if b)
        wrong_conf = [p for p, c in zip(confidence, correct) if not c]
        per_run.append(dict(
            task=d['task'], condition=d['condition'], seed=d['seed'], run_id=d['run_id'],
            source_file=str(path.relative_to(ROOT)), source_sha256=sha(path),
            checkpoint_sha256=d['checkpoint_sha256'], n=n, accuracy=accuracy, nll=nll,
            ece15=ece, binary_brier=statistics.mean((1-p)**2 for p in ptrue),
            mean_confidence=statistics.mean(confidence),
            confidence_on_errors=statistics.mean(wrong_conf),
            wrong_nll=statistics.mean(v for v,c in zip(losses,correct) if not c),
            correct_nll=statistics.mean(v for v,c in zip(losses,correct) if c),
            fraction_zero_loss=sum(v == 0 for v in losses)/n,
            bins=[dict(count=len(b), confidence=statistics.mean(x[0] for x in b) if b else None,
                       accuracy=statistics.mean(x[1] for x in b) if b else None) for b in bins],
        ))
    assert len(per_run) == len(records) == 18
    summary = {}
    for task in ('rte', 'mrpc'):
        summary[task] = {}
        for arm in ('P1_UNREG', 'P1_MIX', 'P1_NORM'):
            rows = [r for r in per_run if (r['task'], r['condition']) == (task, arm)]
            assert sorted(r['seed'] for r in rows) == [17, 42, 123]
            summary[task][arm] = {
                k: dict(mean=statistics.mean(r[k] for r in rows),
                        sample_sd=statistics.stdev(r[k] for r in rows))
                for k in ('accuracy','nll','ece15','binary_brier','mean_confidence',
                          'confidence_on_errors','wrong_nll','correct_nll','fraction_zero_loss')
            }
        summary[task]['mix_minus_norm_per_seed'] = {
            str(seed): {k: next(r[k] for r in per_run if (r['task'],r['condition'],r['seed']) == (task,'P1_MIX',seed))
                        - next(r[k] for r in per_run if (r['task'],r['condition'],r['seed']) == (task,'P1_NORM',seed))
                        for k in ('ece15','binary_brier')}
            for seed in (17,42,123)
        }
    result = dict(
        analysis='Post hoc descriptive calibration; no fitted temperature or model change',
        definitions=dict(ece='15 fixed equal-width confidence bins; seed-level metric then mean over three seeds',
                         brier='Binary Brier mean((p(class 1)-label)^2), not the doubled two-class sum'),
        limitations=['Probabilities reconstructed from saved float32 binary cross-entropy; zero losses saturate at p=1.',
                     'ECE is bin-dependent and noisy on 277/408 fixed examples; these summaries are not independent task replications.',
                     'No inference about temperature-scaled performance or improved representations.'],
        evaluator_sha256=manifest['script_sha256'], source_csv_sha256=sha(DATA/'held_aside_results.csv'),
        summary=summary, runs=per_run,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as handle:
        handle.write(json.dumps(result,indent=2)+'\n')
    for task, arms in summary.items():
        for arm in ('P1_UNREG','P1_MIX','P1_NORM'):
            row=arms[arm]
            print(task,arm,' '.join(f'{k}={row[k]["mean"]:.6f}' for k in ('accuracy','nll','ece15','binary_brier','confidence_on_errors','wrong_nll')))
        print('Paired MIX-NORM:',arms['mix_minus_norm_per_seed'])


if __name__ == '__main__':
    main()
