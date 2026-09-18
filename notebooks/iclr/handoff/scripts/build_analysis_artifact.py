"""Build a sanitized, standalone analysis supplement; never upload or change raw evidence."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BASELINE = '9c8043c1b531f3afb27fa55ce101c8635736c938'
CHECKS = [
    ['check_band_provenance.py'],
    ['build_mixing_tables.py', '--check'], ['build_focused_tables.py', '--check'],
    ['analyze_focused_confirmation.py', '--check'], ['analyze_final_evidence.py', '--check'],
    ['analyze_review_v2.py', '--check'], ['check_spectral_algebra.py'],
    ['analyze_review_v3.py', '--check'],
    ['build_tail_story_tables.py', '--check'],
    ['build_interaction_outcomes_figure.py', '--check'],
    ['check_review_additions.py'], ['audit_manuscript.py'],
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sanitize(text):
    # Preserve recipe/run/step suffixes and every numerical measurement.
    text = re.sub(r'/(?:media|home)/[^\s"\x27,]*?/iclr-campaign-\d+/', 'anonymous_campaign/', text)
    text = re.sub(r'iclr_[0-9a-f]{8}', 'anonymous_study', text)
    text = re.sub(r'6aa54397e58b10444b0fa2aa', 'PRIVATE_PROJECT_ID', text)
    text = re.sub(r'GPU-[0-9a-f-]{36}', 'GPU-REDACTED', text)
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--review', action='store_true',
                        help='Preserve pending author-review markup and audit its allowed locations.')
    args = parser.parse_args()
    checks = [command + (['--review'] if args.review and command[0] == 'audit_manuscript.py' else [])
              for command in CHECKS]
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    originals = {}
    paths = [ROOT/'neurips_2026.tex', ROOT/'reference.bib', ROOT/'LOCKED_EVALUATION_REQUEST_20260916.md',
             ROOT/'archive/retention_2026_09_14.tex']
    paths += list(ROOT.glob('*.sty')) + list(ROOT.glob('*.bst'))
    paths += [p for p in (ROOT/'data').rglob('*') if p.is_file() and p.suffix in ('.csv','.json','.jsonl')]
    paths += [p for p in (ROOT/'scripts').glob('*.py') if p.name not in ('extract_focused_runs.py', Path(__file__).name)]
    paths += list((ROOT/'figures').glob('*.pdf'))
    paths += [p for p in (ROOT/'experiment_source').rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    for src in paths:
        rel = str(src.relative_to(ROOT)); dest = out/rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if src.suffix=='.pdf': dest.write_bytes(src.read_bytes())
        else: dest.write_text(sanitize(src.read_text()))
        originals[rel] = sha(src)
    for name in ('tables/generation.tex','forgetting-results-table.tex','tables/glue.tex'):
        dest=out/'audit_baseline'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(subprocess.check_output(['git','show',f'{BASELINE}:{name}'],cwd=ROOT))

    # Sanitized copies have different bytes. Rebind only the distributed manifest,
    # retaining source hashes in the outer manifest and an explicit transformation note.
    request_path=out/'data/locked_evaluation_request_20260916.json'
    request=json.loads(request_path.read_text())
    population_sha=sha(out/'data/final_evidence_20260916/focused_confirmation_learned.csv')
    request['source_sha256']=population_sha
    request_path.write_text(json.dumps(request,indent=2)+'\n')
    manifest_path=out/'data/locked_evaluation_20260916/evaluation_manifest.json'
    manifest=json.loads(manifest_path.read_text())
    manifest.update(request_markdown_sha256=sha(out/'LOCKED_EVALUATION_REQUEST_20260916.md'),
                    request_json_sha256=sha(request_path),population_source_sha256=population_sha,
                    results_csv_sha256=sha(out/'data/locked_evaluation_20260916/held_aside_results.csv'),
                    artifact_copy_note='Sanitized derivative. These four file hashes bind distributed copies, not the original registration. Original file hashes are in artifact_manifest.json. Checkpoint and validation hashes are unchanged.')
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    audit_path = out/'data/band_source_audit.json'
    audit = json.loads(audit_path.read_text())
    audit['distributed_source_files'] = {name:sha(out/'experiment_source'/name) for name in audit['source_files']}
    audit['artifact_copy_note'] = 'Sanitized derivative. Project identifiers in two source files are anonymized. source_files retains original SHA256/Git blob identities; distributed_source_files hashes the included copies. This is not a claim that anonymized bytes were executed.'
    audit_path.write_text(json.dumps(audit,indent=2,sort_keys=True)+'\n')
    for script in ('analyze_focused_confirmation.py','analyze_final_evidence.py','analyze_review_v2.py','analyze_review_v3.py','build_tail_story_tables.py'):
        subprocess.run([sys.executable,'scripts/'+script,'--write'],cwd=out,check=True)
    assert (out/'neurips_2026.tex').read_text()==sanitize((ROOT/'neurips_2026.tex').read_text())
    for command in checks:
        subprocess.run([sys.executable,'scripts/'+command[0],*command[1:]],cwd=out,check=True)
    for cache in out.rglob('__pycache__'):
        for f in cache.iterdir(): f.unlink()
        cache.rmdir()
    # No raw data text or private collaboration documents are included.
    for p in out.rglob('*'):
        if p.is_file() and p.suffix!='.pdf':
            text=p.read_text()
            assert not re.search(r'guyb|/media/|6aa54397|ortho_new|GPU-[0-9a-f-]{36}|olp_[A-Za-z0-9]+',text),p
    (out/'requirements.txt').write_text(''.join(f'{name}=={importlib.metadata.version(name)}\n' for name in ('numpy','scipy','matplotlib','PyMuPDF')))
    readme = '''# Supplementary analysis archive

Run `python3 verify_artifact.py` from this directory after installing requirements.txt.
The checks use CPU only, do not fetch remote data, and do not require Git history.
Optional PDF build: `tectonic -X compile neurips_2026.tex --keep-logs`;
then `python3 scripts/audit_manuscript.py --pdf neurips_2026.pdf`.

Contents: paper sources/styles, exported measurements, thirteen distinct-recipe
manifests, selection frontier, invalidation/preparation records, 18 held-aside
per-example prediction/label/loss records, analysis/regeneration scripts, figures,
and hash-checked historical table fixtures. The GLUE CSV preserves 60 original
means/SDs; imported rows use the RandLoRA protocol, spectral rows use the historical
protocol with unresolved run/count provenance. MRPC in that table is accuracy.
The inactive retention table is an exclusion/provenance record, not paper evidence.
All 21 registered RTE band/head confirmations are included with their freeze
ledger, plus two separately labeled timing pilots. Band scores are inner-selection
scores; official held-aside scoring of this block is absent from this release.
Do not pool pilots with confirmations or band scores with the separate 18-run
practical held-aside evaluation.

This is a sanitized derivative, not the untouched server export. Private path
prefixes, project labels and GPU identifiers are replaced. Run IDs, seeds, steps,
numerical measurements, source revisions, and checkpoint/validation hashes remain.
artifact_manifest.json records both source and distributed hashes for copied files.
The distributed held-aside request/manifest has hashes rebound to sanitized copies;
it does not constitute a new preregistration. Generated analysis hashes refer to
distributed bytes. SHA checks establish consistency, not independent authenticity.

The experiment_source directory includes the 46 recorded Python source files,
including the training runner and its tests. The ten recorded band-run source
revisions have identical campaign trees, and their source fingerprints were
checked against Git blobs. The two files containing a private project identifier
are anonymized in this archive; both original and distributed hashes are retained.
See experiment_source/README.md for the scope of this source snapshot.

No model weights, raw dataset text, original validation reports,
full protocol-registration documents, or complete historical run manifests are
included. Checkpoint hashes identify unavailable upstream artifacts;
they do not make those artifacts downloadable. These scripts reproduce analysis
of the exports, not training, checkpoint validation, or independently timed execution.
Centered-scaler experiments, temperature-scaling logits and full training curves
are not included. No claim about their outcomes is made.
This archive is prepared for author delivery as supplementary material; it does
not assert that a public repository or anonymous hosting endpoint already exists.
'''
    if args.review:
        readme = readme.replace('scripts/audit_manuscript.py --pdf',
                                'scripts/audit_manuscript.py --review --pdf')
        readme += '\nPaper sources retain pending blue/gray author-review markup. The audit allows it only in the designated review sections.\n'
    (out/'README.md').write_text(readme)
    verify = 'import hashlib,json,subprocess,sys\nfrom pathlib import Path\nroot=Path(__file__).resolve().parent\nm=json.loads((root/"artifact_manifest.json").read_text())\nfor name,item in m["files"].items():\n assert hashlib.sha256((root/name).read_bytes()).hexdigest()==item["distributed_sha256"],name\n'
    verify += f'for command in {checks!r}:\n subprocess.run([sys.executable,"scripts/"+command[0],*command[1:]],cwd=root,check=True)\nprint("PASS: distributed hashes and all analysis checks.")\n'
    (out/'verify_artifact.py').write_text(verify)
    files={str(p.relative_to(out)):dict(source_sha256=originals.get(str(p.relative_to(out))),distributed_sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file()}
    (out/'artifact_manifest.json').write_text(json.dumps(dict(scope='Sanitized analysis supplement; source hashes identify original private bytes, not included duplicates.',files=files),indent=2)+'\n')
    with zipfile.ZipFile(out.with_suffix('.zip'),'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file(): z.write(p,p.relative_to(out))
    print('PASS: artifact',out.with_suffix('.zip'),'SHA256',sha(out.with_suffix('.zip')))


if __name__=='__main__':
    main()
