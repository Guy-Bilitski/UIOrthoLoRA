"""CPU checks for review-driven algebra/reporting, not new trained-model results."""
from collections import defaultdict
from statistics import stdev
import numpy as np

from analyze_archived_layers import analyze, records, key
from build_mixing_tables import reconstruct

rng = np.random.default_rng(20260914)
close = lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-10)

for d, k in ((4, 1), (7, 3), (9, 7)):
    eps = rng.normal(size=d)
    expected = k*(d-k)/((d-1)*(d+2))*np.sum((eps-eps.mean())**2)
    observed = []
    for _ in range(6000):
        q = np.linalg.qr(rng.normal(size=(d, d)))[0]
        p = q[:, :k] @ q[:, :k].T
        w = p*p
        lap = np.diag(w.sum(axis=1))-w
        close(lap @ np.ones(d), np.zeros(d))
        j = eps @ lap @ eps
        close((eps+2.5) @ lap @ (eps+2.5), j)
        close(np.sum(-2*lap @ eps), 0)
        observed.append(j)
    se = np.std(observed, ddof=1)/np.sqrt(len(observed))
    assert abs(np.mean(observed)-expected) < 5*se
    print(f'Haar projector expectation d={d}, k={k}: error={abs(np.mean(observed)-expected):.4g}, MC SE={se:.4g}')

    # Chordal normalization for all possible intersection dimensions.
    a = np.linalg.qr(rng.normal(size=(d, d)))[0][:, :k]
    b = np.linalg.qr(rng.normal(size=(d, d)))[0][:, :k]
    p, pp = a @ a.T, b @ b.T
    chord2 = np.sum((p-pp)**2)/(2*min(k, d-k))
    overlap = np.trace(p @ pp)/k
    close(chord2, k/min(k, d-k)*(1-overlap))
    assert 0 <= chord2 <= 1+1e-10
    c = rng.normal(size=(k, k))
    identity = np.trace(c)/k*np.eye(k)
    eta = np.trace(c)**2/(k*np.sum(c*c))
    close(np.sum(identity**2), eta*np.sum(c*c))
    close(np.sum(c*c), np.sum(identity**2)+np.sum((c-identity)**2))
    assert 0 <= eta <= 1

for s, h in ((.1, .1), (.01, .1), (.01, .01)):
    diagonal = s*s*np.r_[np.ones(512), h*np.ones(256)]
    close(np.linalg.norm(diagonal), s*s*np.sqrt(512+256*h*h))
    close(np.sum(diagonal[:512]**2)/np.sum(diagonal**2), 512/(512+256*h*h))

_, _, alloc = analyze(include_runs=True)
rows = {key(r): r for r in records('legacy_mixing')}
assert set(rows) == set(alloc) and len(rows) == 27
grouped = defaultdict(list)
for run, row in rows.items():
    assert int(row['trainable_params']) == 86016
    score = float(row.get('final_val_score', row.get('final_val_accuracy')))
    grouped[run[:2]].append((score*100, float(row['mean_RelPert_F']), 100*alloc[run]['pCross']))
    assert float(row['total_train_time']) > 0 and float(row['peak_gpu_memory']) > 0
for group, values in grouped.items():
    if len(values) == 2:
        for v in zip(*values):
            close(stdev(v), abs(v[0]-v[1])/np.sqrt(2))
    else:
        assert len(values) == 1 and group[0] == 'sst2_lin'

generated = reconstruct()
for label, count in (('MIXING SUMMARY', 15), ('MIXING FULL', 27), ('ARCHIVED COST', 27)):
    data = [line for line in generated[label].splitlines() if ' & ' in line]
    assert len(data) == count, (label, len(data))
print(f'Archived training-call total: {sum(float(r["total_train_time"]) for r in rows.values())/3600:.4f} hours (not total project GPU-hours).')
print('PASS: projector expectation/mean conservation, chordal and identity diagnostics, initialization norms, seed SDs, all 27 cost rows.')
