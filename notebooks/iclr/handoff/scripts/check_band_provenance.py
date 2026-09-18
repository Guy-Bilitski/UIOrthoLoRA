"""Verify exported band checkpoints and the recorded experiment-source identity.

The optional --record mode reads a separate research Git checkout. Offline checks
reproduce joins and verify the saved source snapshot; they do not reload models.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/final_evidence_20260916'
AUDIT = ROOT / 'data/band_source_audit.json'
SNAPSHOT = ROOT / 'experiment_source'
PREFIX = 'notebooks/iclr/campaign'


def inputs():
    rows = list(csv.DictReader((DATA/'band_results.csv').open()))
    manifests = [json.loads(p.read_text()) for p in (DATA/'manifests').glob('*.json')]
    recipes = {(m['task'], m['condition']):m for m in manifests}
    return rows, recipes


def record(repo, archive):
    rows, recipes = inputs()
    revisions = sorted({r['source_revision'] for r in rows} |
                       {recipes[r['task'],r['condition']]['source_revision'] for r in rows})
    trees = {}
    for revision in revisions:
        raw = subprocess.check_output(['git','ls-tree','-r',revision,'--',PREFIX], cwd=repo, text=True)
        trees[revision] = {line.split('\t')[1]:line.split()[2] for line in raw.splitlines()}
    reference = recipes['rte','BAND_TAIL_DIAG']['source_revision']
    assert all(tree == trees[reference] for tree in trees.values()), 'Experiment code differs between revisions'
    expected = recipes['rte','BAND_TAIL_DIAG']['source_files_sha256']
    assert all(recipes[r['task'],r['condition']]['source_files_sha256'] == expected for r in rows)
    files = {}
    with tarfile.open(archive) as tar:
        for name, expected_sha in sorted(expected.items()):
            assert name in trees[reference]
            content = tar.extractfile(name).read()
            assert hashlib.sha256(content).hexdigest() == expected_sha, name
            blob = hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
            assert blob == trees[reference][name], name
            target = SNAPSHOT/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            files[name] = dict(sha256=expected_sha, git_blob=blob)
    result = dict(
        scope='Read-only Git tree comparison across all recorded band-run revisions; recorded source fingerprints checked against exact Git blobs. Offline checks verify the saved snapshot and exported checkpoint joins, not execution or checkpoint reload.',
        reference_revision=reference, campaign_trees=trees, source_files=files,
    )
    AUDIT.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')


def check():
    rows, recipes = inputs()
    audit = json.loads(AUDIT.read_text())
    trees = audit['campaign_trees']
    reference = trees[audit['reference_revision']]
    assert reference and all(tree == reference for tree in trees.values())
    expected = {name:item['sha256'] for name,item in audit['source_files'].items()}
    distributed = audit.get('distributed_source_files')
    if distributed is not None:
        assert set(distributed) == set(expected)
        assert audit['artifact_copy_note'].startswith('Sanitized derivative.')
    for name, item in audit['source_files'].items():
        content = (SNAPSHOT/name).read_bytes()
        actual_sha = hashlib.sha256(content).hexdigest()
        assert actual_sha == (distributed[name] if distributed else item['sha256']), name
        assert item['git_blob'] == reference[name], name
        if actual_sha == item['sha256']:
            assert hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest() == item['git_blob'], name
    for r in rows:
        recipe = recipes[r['task'],r['condition']]
        assert recipe['source_files_sha256'] == expected
        assert r['source_revision'] in trees and recipe['source_revision'] in trees
    coverage = json.loads((DATA/'band_coverage.json').read_text())
    freeze = [json.loads(line) for line in (ROOT/'data/locked_evaluation_20260916/block_b_freeze.jsonl').read_text().splitlines()]
    confirmations = {r['run_id']:r for r in rows if r['purpose']=='confirmation'}
    assert len(freeze) == len(confirmations) == 21
    assert {f['run_id'] for f in freeze} == set(confirmations)
    assert len({(f['task'],f['condition'],f['seed']) for f in freeze}) == 21
    for f in freeze:
        r = confirmations[f['run_id']]
        assert all(str(f[k]) == r[k] for k in ('task','condition','seed','source_revision'))
        assert f['validation_report_sha256'] == r['validation_sha256']
        assert f['protocol_sha256'] == coverage['protocol_sha256']
        assert f['fixed_optimizer_step'] == int(r['fixed_steps']) == 5670
        assert f['run_id'] in Path(f['fixed_checkpoint']).parts
        assert Path(f['fixed_checkpoint']).name == 'step_00005670'
        assert len(f['checkpoint_sha256']) == 64
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', action='store_true')
    parser.add_argument('--research-repo', type=Path)
    parser.add_argument('--archive', type=Path)
    args = parser.parse_args()
    if args.record:
        assert args.research_repo and args.archive
        record(args.research_repo, args.archive)
    audit = check()
    kind = 'distributed source hashes checked; original fingerprints retained' if audit.get('distributed_source_files') else 'source files match manifest SHA256 and Git blobs'
    print(f'PASS: 21 frozen confirmations joined; {len(audit["campaign_trees"])} recorded source revisions have identical campaign trees; {len(audit["source_files"])} {kind}.')


if __name__ == '__main__':
    main()
