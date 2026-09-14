"""Small synthetic consistency checks for the manuscript identities, not model experiments."""
import numpy as np

rng = np.random.default_rng(20270913)
op = lambda x: np.linalg.norm(x, 2)
sqf = lambda x: float(np.sum(x*x))

def close(a, b):
    np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-10)

def check_square(d, k):
    u = np.linalg.qr(rng.normal(size=(d, d)))[0]
    v = np.linalg.qr(rng.normal(size=(d, d)))[0]
    ul, ut, vl, vt = u[:, :k], u[:, k:], v[:, :k], v[:, k:]
    g = rng.normal(size=(d, d))
    gtt = ut.T @ g @ vt
    close(sqf(g-ut @ gtt @ vt.T), sqf(g)-sqf(gtt))
    close(sqf(g-ut @ np.diag(np.diag(gtt)) @ vt.T), sqf(g)-sqf(np.diag(gtt)))
    # Full-core inner SVD, rank-constrained projection, and basis invariance.
    left, values, right = np.linalg.svd(gtt)
    close(left @ np.diag(values) @ right, gtt)
    for q in range(d-k+1):
        truncated = (left[:, :q] * values[:q]) @ right[:q, :]
        close(sqf(g-ut @ truncated @ vt.T), sqf(g)-sqf(values[:q]))
    rt = np.linalg.qr(rng.normal(size=(d-k,d-k)))[0]
    close(sqf(gtt), sqf(rt.T @ gtt @ rt))
    for s in (np.zeros((k,k)), np.eye(k), rng.normal(size=(k,k))):
        e = np.diag(rng.normal(size=d))
        ds = np.diag(rng.normal(size=d))
        h = rng.normal(size=(d-k,d-k))
        inner = ul @ s @ vl.T + ut @ h @ vt.T
        delta = e @ inner @ ds
        c = u.T @ delta @ v
        a, b = u.T @ e @ u, v.T @ ds @ v
        p = ul @ ul.T
        weights = p*p
        eps = np.diag(e)
        close(sqf(a[:k,k:]), 0.5*sqf(e @ p-p @ e))
        close(sqf(a[:k,k:]), 0.5*np.sum(weights*(eps[:,None]-eps[None,:])**2))
        laplacian = np.diag(weights.sum(axis=1)) - weights
        close(sqf(a[:k,k:]), eps @ laplacian @ eps)
        close(laplacian @ np.ones(d), 0)
        blocks = [c[:k,:k], c[:k,k:], c[k:,:k], c[k:,k:]]
        close(sum(sqf(x) for x in blocks), sqf(delta))
        core = np.zeros((d,d)); core[:k,:k] = s; core[k:,k:] = h
        close(c, a @ core @ b)
        mu, nu, en, dn, sn, hn = op(a[:k,k:]), op(b[:k,k:]), op(e), op(ds), op(s), op(h)
        bounds = [en*sn*dn+mu*hn*nu, en*sn*nu+mu*hn*dn,
                  mu*sn*dn+en*hn*nu, mu*sn*nu+en*hn*dn]
        assert all(op(x) <= bound + 1e-10 for x, bound in zip(blocks, bounds))
        cross = c.copy(); cross[:k,:k] = 0; cross[k:,k:] = 0
        close(op(cross), max(op(blocks[1]), op(blocks[2])))
        if sn == 0:
            off = c.copy(); off[k:,k:] = 0
            assert op(off) <= hn*np.sqrt((mu*nu)**2+(mu*dn)**2+(en*nu)**2)+1e-10
            scale = 0.2
            close((scale*e) @ ut @ (h/scale**2) @ vt.T @ (scale*ds), delta)
        close((0.2*e) @ inner @ (ds/0.2), delta)
    # Identity scalers: no cross blocks, with a possibly nonzero leading core.
    c = u.T @ (ul @ vl.T) @ v
    close(c[:k,:k], np.eye(k)); close(c[:k,k:], 0); close(c[k:,:k], 0)

def check_rectangular(m, n, k):
    u = np.linalg.qr(rng.normal(size=(m,m)))[0][:,:k]
    v = np.linalg.qr(rng.normal(size=(n,n)))[0][:,:k]
    p, q = u @ u.T, v @ v.T
    delta = rng.normal(size=(m,n))
    blocks = [a @ delta @ b for a in (p,np.eye(m)-p) for b in (q,np.eye(n)-q)]
    close(sum(blocks), delta); close(sum(sqf(x) for x in blocks), sqf(delta))

square_cases = rectangular_cases = 0
for d in (2, 3, 5, 8, 12):
    for k in range(1, d):
        for _ in range(4):
            check_square(d, k)
            square_cases += 1
for m, n in ((1, 4), (4, 1), (3, 7), (7, 3), (5, 5)):
    for k in range(min(m, n)+1):
        for _ in range(4):
            check_rectangular(m, n, k)
            rectangular_cases += 1

# Coordinate-aligned and disconnected projectors: the nullspace need not be scalar.
for p, e in ((np.diag([1., 0., 1.]), np.diag([2., 3., 4.])),
             (np.array([[.5, .5, 0], [.5, .5, 0], [0, 0, 1.]]), np.diag([2., 2., 5.]))):
    close(e @ p - p @ e, 0)

# Zero update has zero block energies, but normalized fractions are undefined.
close(sqf(np.zeros((4, 4))), 0)

base = np.diag([2.,1.]); adapted = base + np.diag([0.,2.])
ub = np.linalg.svd(base)[0][:,0]; ua = np.linalg.svd(adapted)[0][:,0]
close(abs(ub @ ua), 0)
print(f'PASS: {square_cases} projection/core tests, {3*square_cases} unified block-bound and commutator cases, {rectangular_cases} rectangular decompositions, factor symmetries, and boundary counterexamples.')
