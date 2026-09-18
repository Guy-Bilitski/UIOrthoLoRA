"""CPU checks for review-driven algebra/reporting, not new trained-model results."""
from collections import defaultdict
from statistics import stdev
import numpy as np

from analyze_archived_layers import analyze, records, key
from build_mixing_tables import reconstruct

rng = np.random.default_rng(20260914)
close = lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-10)

# New coordinate-table claims: pure left/right orthogonal maps preserve singular
# values without requiring support in a selected pretrained span. LoRA-XS retains
# Sigma in the literal published left factor. These are matrix identities only.
for d in (4, 7, 11):
    u=np.linalg.qr(rng.normal(size=(d,d)))[0]
    v=np.linalg.qr(rng.normal(size=(d,d)))[0]
    sigma=np.diag(np.arange(d,0,-1,dtype=float))
    w=u@sigma@v.T
    rotation=np.linalg.qr(rng.normal(size=(d,d)))[0]
    close(u.T@(rotation@w-w)@v,(u.T@rotation@u-np.eye(d))@sigma)
    close(u.T@(w@rotation-w)@v,sigma@(v.T@rotation@v-np.eye(d)))
    close(np.linalg.svd(rotation@w,compute_uv=False),np.diag(sigma))
    close(np.linalg.svd(w@rotation,compute_uv=False),np.diag(sigma))
    j=np.arange(d-2,d);core=rng.normal(size=(2,2))
    delta=u[:,j]@sigma[np.ix_(j,j)]@core@v[:,j].T
    c=np.zeros((d,d));c[np.ix_(j,j)]=sigma[np.ix_(j,j)]@core
    close(u.T@delta@v,c)
    e=np.diag(rng.normal(size=d));f=np.diag(rng.normal(size=d))
    delta=e@w@f;t=.3;scaled=(t*e)@w@(t*f)
    close(scaled,t*t*delta)
    close(np.sum(scaled**2),t**4*np.sum(delta**2))
print('PASS: prior-method coordinate identities, pure orthogonal spectra, and fixed-core joint scaling.')

# PSOFT's internal vector relaxation keeps principal support while it can change
# Gram geometry. Fixed-support and isometry are separate properties.
psrng=np.random.default_rng(20260916)
u=np.linalg.qr(psrng.normal(size=(7,7)))[0];v=np.linalg.qr(psrng.normal(size=(7,7)))[0]
sig=np.diag([7.,5.,3.]);rot=np.linalg.qr(psrng.normal(size=(3,3)))[0]
a=u[:,:3];b=sig@v[:,:3].T
for core in (rot,np.diag([1.,2.,.7])@rot@np.diag([.8,1.2,1.])):
    delta=a@(core-np.eye(3))@b
    c=u.T@delta@v;expected=np.zeros((7,7));expected[:3,:3]=(core-np.eye(3))@sig
    close(c,expected)
close((a@rot@b).T@(a@rot@b),(a@b).T@(a@b))
assert not np.allclose((a@core@b).T@(a@core@b),(a@b).T@(a@b))

# Explicit counterexample to radial effective-update flow under a factor penalty.
angle=np.pi/6;u=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
fractions=[]
for h in (1.,.5,0.):
    c=u.T@np.diag([2.,1.])@u@np.diag([1.,h])
    close(np.sum(c*c),(13+7*h*h)/4)
    f=(c[0,1]**2+c[1,0]**2)/np.sum(c*c)
    close(f,3*(1+h*h)/(4*(13+7*h*h)));fractions.append(f)
assert fractions[0]>fractions[1]>fractions[2]
print('PASS: PSOFT support/Gram distinction and nonradial factor-parameterized magnitude counterexample.')

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
