"""
CorDA++ init  (arXiv:2506.13187, "dynamic Context-oriented Decomposition") for our shared LoRA trainer.

CorDA++ = static CorDA-KPM (knowledge-preserved: adapter = the SMALLEST-r context-oriented singular
directions, big/knowledge directions frozen in the residual) + two dynamic ingredients:
  (1) DYNAMIC COVARIANCE SELECTION (Eq 7-8): for each target layer, sample N candidate calibration
      covariances and pick, PER LAYER, the one whose context-oriented decomposition is most compact
      + most spectrally concentrated (argmin of a compactness x tail score).
  (2) DYNAMIC RANK ALLOCATION (Eq 9-10): start every layer at rank 1 and greedily grow the
      lowest-scoring layer's rank by 1 until a global param budget tau is exceeded (overshoot by one).
      Ranks are redistributed across layers; the total trainable-param count is ~param-matched to a
      fixed-rank arm (r=64 math / r=32 CS).

This module owns ONLY the CorDA++-specific math + injection. It REUSES the faithful static
context-oriented decomposition from corda_init.py (damped inverse of C; Wc=W@C_fix; SVD; V=(Vt@C_inv)^T;
KPM = bottom-r), so the injected adapter is numerically identical to our validated CorDA path at each
layer's allocated rank. Loss-preserving: W_res + scaling*B@A == W0 (residual method -> residual_save).

--------------------------------------------------------------------------------------------------
DESIGN DECISIONS / FAITHFULNESS CAVEATS (see handoff/17 §3, §8):
  * pi(C) OPERAND (handoff/17 §3.1 ambiguity): the paper writes pi(C)=sqrt(d_out*sigma_max)/sigma_min
    "in terms of sigma(C)". We implement compactness() on the singular values of C itself by default
    (this is what the CPU validator checks against the closed form), and expose an `svals=` override so a
    caller can instead feed the singular values of the context-oriented product W@C_fix if a paper fetch
    resolves the operand that way. Selection/allocation call compactness() on C (literal form); the tail
    terms sigma_{-r} always come from the decomposition of W@C_fix (unambiguous in the paper).
  * RANK-ALLOCATION DIRECTION (Eq 9-10): the literal score s^l = log(pi^l) * sigma_{-r^l} / sum_top is
    strictly INCREASING in pi (log(pi) enters with a +sign) and the greedy increments the argMIN. Holding
    the spectrum fixed, this gives MORE rank to LOWER-pi (== "more compact", well-conditioned) layers and
    LESS rank to higher-pi ("less compact") layers. This matches the handoff's own reading of the twin
    selection score (§3.2: argmin "favors compact, small-pi"). The task note "more rank to less-compact"
    reads inverted vs the literal formula + the selection interpretation; we implement the FORMULA and the
    CPU validator asserts+prints the realized direction. Flagged as an open item.
  * N (candidate pool size): NOT published (handoff/17 §8 fetch-blocker). Exposed as a CLI knob
    (--cordapp_n) with a documented default DEFAULT_N=8. Numbers built on this default are pipeline-valid
    but NOT a faithful reproduction of the paper's N until fetched.
  * SCALING: to keep parity with the fixed-rank arms (alpha/r = 128/64 = 64/32 = 2) at EVERY layer
    despite per-layer ranks, the caller sets alpha_pattern[l] = 2*r^l so scaling^l = 2 everywhere. The
    KPM init folds scaling into B (B/scaling), so it is loss-preserving for any scaling (residual_save
    generalized). "scaling=2 (alpha=2r)" is the faithful setting the CPU validator checks.
--------------------------------------------------------------------------------------------------
"""
import math
import torch
import torch.nn as nn
from peft.tuners.tuners_utils import BaseTunerLayer

import corda_init  # faithful static context-oriented decomposition (reused for the injected init)

DEFAULT_N = 8  # candidate covariance pool size; paper value unresolved (handoff/17 §8) -> documented default


# ============================== context-oriented decomposition ==============================
# NOTE: this damped-inverse + covariance-oriented SVD MIRRORS corda_init.corda_kpa_BAR exactly
# (keep in sync). It differs only in that it returns the FULL spectrum (needed for the CorDA++ scores),
# whereas corda_kpa_BAR returns the r-sliced factors. kpm_BAR() below builds the same B/A/W_res from it,
# and validate_cordapp_cpu.py asserts kpm_BAR == corda_init.corda_kpa_BAR byte-for-byte.
@torch.no_grad()
def context_decompose(W, C):
    """Context-oriented decomposition of W under input covariance C.
    Returns (U[out,k], S[k] DESCENDING, V[in,k], C_inv[in,in]); Wc = W@C_fix = U diag(S) (V^T C_fix)."""
    W = W.float(); C = C.float().to(W.device); insz = W.shape[1]
    I = torch.eye(insz, device=W.device, dtype=W.dtype)
    damp = 0.01
    mdiag = torch.diag(C).mean()
    for _ in range(20):
        C_fix = C + damp * mdiag * I
        C_inv = torch.linalg.inv(C_fix)
        if torch.linalg.matrix_norm(C_fix @ C_inv - I, ord=2) < 0.05:
            break
        damp *= 2
    Wc = W @ C_fix
    U, S, Vt = torch.linalg.svd(Wc, full_matrices=False)   # U(out,k) S(k) Vt(k,in), S descending
    V = (Vt @ C_inv).transpose(-1, -2)                     # (in,k) — undo covariance on the right factor
    return U, S, V, C_inv


@torch.no_grad()
def kpm_BAR(W, C, r):
    """KPM (== corda_init KPA): bottom-r context directions become the trainable adapter.
    Returns (B[out,r], A[r,in], W_res[out,in]); loss-preserving: W_res + B@A == W."""
    U, S, V, _ = context_decompose(W, C)
    Ur, Sr, Vr = U[:, -r:], S[-r:], V[:, -r:]              # KPM: smallest r
    B = Ur * Sr.sqrt()                                     # (out,r)
    A = Sr.sqrt()[:, None] * Vr.transpose(-1, -2)          # (r,in)
    W_res = W - B @ A
    return B, A, W_res


# ============================== compactness metric ==============================
@torch.no_grad()
def compactness(C, d_out, svals=None):
    """pi(C) = sqrt(d_out * sigma_max) / sigma_min   (lower = more compact / better conditioned).

    By default sigma_* are the singular values of the covariance C (== its eigenvalues when C is SPD) —
    the literal Eq form the CPU validator checks. Pass svals=<descending spectrum of W@C_fix> to compute
    pi on the context-oriented product instead (handoff/17 §3.1 operand ambiguity)."""
    s = torch.linalg.svdvals(C.float()) if svals is None else svals.float()
    return math.sqrt(float(d_out) * float(s.max())) / float(s.min())


# ============================== candidate covariance collection ==============================
def _calib_rounds(prompts, N, calib_size):
    """Partition the first `calib_size` prompts into <=N disjoint contiguous rounds (the pool D={I_1..I_N};
    one covariance per sampling round)."""
    prompts = list(prompts)[:calib_size]
    n_rounds = max(1, N)
    sz = max(1, math.ceil(len(prompts) / n_rounds))
    rounds = [prompts[k * sz:(k + 1) * sz] for k in range(n_rounds)]
    return [r for r in rounds if r]                       # drop empties if calib_size < N


def _target_linears(model, targets):
    tset = tuple(targets)
    return [(n, m) for n, m in model.named_modules()
            if isinstance(m, nn.Linear) and any(n == t or n.endswith("." + t) for t in tset)]


@torch.no_grad()
def _round_cov(model, round_prompts, tokenizer, targets, max_len, bs):
    """One sampling round's CorDA-normalized covariance per target Linear (x/max|x|, then X^T X, divided
    by contributing-sample count). Mirrors corda_init.collect_corda_cov normalization exactly."""
    cur, hooks = {}, []

    def mk(name):
        def h(mod, inp):
            x = inp[0].reshape(-1, inp[0].shape[-1]).float()
            m = x.max().abs()
            if m > 0:
                x = x / m
            c = x.transpose(-1, -2) @ x
            if torch.isfinite(c).all():
                cur[name] = cur.get(name, 0.0) + c
        return h

    for n, mod in _target_linears(model, targets):
        hooks.append(mod.register_forward_pre_hook(mk(n)))
    try:
        cnt = 0
        for i in range(0, len(round_prompts), bs):
            batch = round_prompts[i:i + bs]
            enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                            max_length=max_len).to(model.device)
            model(**enc)
            cnt += len(batch)
    finally:
        for h in hooks:
            h.remove()
    return {name: c / max(1, cnt) for name, c in cur.items()}


@torch.no_grad()
def collect_candidate_covs(model, prompts, tokenizer, targets, N=DEFAULT_N,
                           calib_size=256, max_len=256, bs=4):
    """REFERENCE (non-streaming) candidate collection: {fully_qualified_name: [C_1, ..., C_N]} (fp32, CPU),
    one covariance per disjoint calib round. Keys are the raw module names -> use as rank_pattern keys
    (handoff/17 FIX-3: suffix-matched by PEFT get_pattern_key).

    NOTE: materializes all N covariances for every layer at once. Fine for small models / validation; for
    a 7B model this is O(N * sum_l in_l^2) (~190GB at N=8) -> use precompute_cordapp() which STREAMS
    round-by-round and keeps only the best cov per layer."""
    cands = {}
    for rp in _calib_rounds(prompts, N, calib_size):
        rc = _round_cov(model, rp, tokenizer, targets, max_len, bs)
        for name, c in rc.items():
            cands.setdefault(name, []).append(c.detach().cpu())
    return cands


# ============================== dynamic covariance selection (Eq 7-8) ==============================
@torch.no_grad()
def cov_selection_score(C, W):
    """s(C) = log(pi(C)) * sum_{r=1..R}(sigma_{-r}/sigma_max), sigma from the decomposition of W@C_fix.
    sum_{r=1..R}(sigma_{-r}/sigma_max) == (sum of all sigma)/sigma_max (a spectral-flatness measure;
    small == concentrated). Lower score wins. Returns (score, S, pi) so S/pi can be reused downstream."""
    _, S, _, _ = context_decompose(W, C)
    d_out = W.shape[0]
    pi = compactness(C, d_out)
    tail = float(S.sum() / S[0])           # sum_{r=1..R} sigma_{-r}/sigma_max
    return math.log(pi) * tail, S, pi


@torch.no_grad()
def select_covariances(cands, weights):
    """Per layer, argmin over the N candidates of cov_selection_score (Eq 8), independently per layer.
    cands: {name: [C_i]}, weights: {name: W[out,in]}.
    Returns (chosen={name: C*}, spectra={name: S* of W@C*_fix}, info={name: {chosen_idx, score, pi, ...}})."""
    chosen, spectra, info = {}, {}, {}
    for name, cov_list in cands.items():
        if name not in weights:
            continue
        W = weights[name]
        best_i, best_s, best_S, best_pi = -1, math.inf, None, None
        for i, C in enumerate(cov_list):
            s, S, pi = cov_selection_score(C, W)
            if s < best_s:
                best_i, best_s, best_S, best_pi = i, s, S, pi
        chosen[name] = cov_list[best_i]
        spectra[name] = best_S
        info[name] = {"chosen_idx": best_i, "score": best_s, "pi": best_pi, "n_cands": len(cov_list)}
    return chosen, spectra, info


# ============================== dynamic rank allocation (Eq 9-10, KPM) ==============================
@torch.no_grad()
def alloc_score(pi, S, r):
    """KPM allocation score at current rank r (Eq 10):
        s = log(pi) * sigma_{-r} / sum_{k=1..(R-r)} sigma_k
    numerator = r-th smallest sigma (== S[-r]); denominator = the top (R-r) sigma (the frozen residual
    energy after r bottom directions are peeled into the adapter). Strictly increasing in pi -> the
    greedy argMIN grows lower-pi (more compact) layers faster. r must satisfy 1 <= r < R."""
    R = S.shape[0]
    num = float(S[-r])
    den = float(S[:R - r].sum())
    return math.log(pi) * num / den


@torch.no_grad()
def allocate_ranks(chosen_covs, spectra, weights, tau, start_rank=1, pis=None):
    """Greedy per-layer rank allocation to a global trainable-param budget tau (Eq 9-10, KPM).

    tau counts (d_in + d_out)*r per layer (LoRA A[r,in]+B[out,r] param count). Start r^l=start_rank
    everywhere, repeatedly increment the argMIN-scoring layer, stop at the FIRST overshoot (tau' > tau)
    keeping that last increment (so realized tau' overshoots by <= one (d_in+d_out) step).

    `pis` (optional): precomputed {name: pi} to avoid re-running svdvals(C) (a large SVD for e.g.
    down_proj); if None, compactness(C) is computed per layer. Returns (ranks={name: r^l}, realized_tau,
    meta) where meta[name]={pi, R, d_in, d_out}."""
    names = [n for n in chosen_covs if n in spectra and n in weights]
    dims, pi_map, Rs = {}, {}, {}
    for n in names:
        W = weights[n]
        d_out, d_in = int(W.shape[0]), int(W.shape[1])
        dims[n] = (d_in, d_out)
        Rs[n] = int(spectra[n].shape[0])
        pi_map[n] = pis[n] if (pis is not None and n in pis) else compactness(chosen_covs[n], d_out)
    pis = pi_map
    ranks = {n: min(start_rank, Rs[n]) for n in names}

    def realized():
        return sum((dims[n][0] + dims[n][1]) * ranks[n] for n in names)

    while True:
        best_n, best_s = None, math.inf
        for n in names:
            r = ranks[n]
            if r >= Rs[n]:            # layer saturated at full rank
                continue
            s = alloc_score(pis[n], spectra[n], r)
            if s < best_s:
                best_n, best_s = n, s
        if best_n is None:            # every layer saturated before hitting tau
            break
        ranks[best_n] += 1
        if realized() > tau:          # overshoot by one -> stop, keep the increment
            break

    meta = {n: {"pi": pis[n], "R": Rs[n], "d_in": dims[n][0], "d_out": dims[n][1]} for n in names}
    return ranks, realized(), meta


def budget_tau(weights, fixed_rank):
    """Global budget = param-match to a fixed-rank arm: tau = fixed_rank * sum_l (d_in^l + d_out^l)."""
    return int(fixed_rank) * sum(int(W.shape[0]) + int(W.shape[1]) for W in weights.values())


def build_patterns(ranks, scaling=2.0):
    """Build LoraConfig pattern dicts (fully-qualified keys). rank_pattern[l]=r^l; alpha_pattern[l] set so
    scaling^l = alpha^l/r^l = `scaling` at EVERY layer (parity with the fixed-rank arms, s=2)."""
    rank_pattern = {n: int(r) for n, r in ranks.items()}
    alpha_pattern = {n: int(round(scaling * int(r))) for n, r in ranks.items()}
    return rank_pattern, alpha_pattern


# ============================== injection (KPM init at per-layer rank) ==============================
def _resolve_covkey(wrapped_name, cov_keys):
    """Match a PEFT-wrapped module name (e.g. 'base_model.model.model.layers.0.self_attn.q_proj') to the
    RAW cov key ('model.layers.0.self_attn.q_proj') by exact-or-suffix match (keys are unique full paths)."""
    if wrapped_name in cov_keys:
        return wrapped_name
    for k in cov_keys:
        if wrapped_name == k or wrapped_name.endswith("." + k):
            return k
    return None


@torch.no_grad()
def apply_cordapp(peft_model, chosen_covs, ranks=None, adapter="default"):
    """Inject KPM init at each layer's ALLOCATED rank (read from the built LoRA layer's own r, which
    PEFT resolved from rank_pattern). Folds scaling into B (B/scaling) and overwrites base.weight=W_res,
    exactly like corda_init.apply_corda -> residual_save compatible. Returns max reconstruction error
    max|W_res + scaling*B@A - W0| (~0: loss-preserving). If `ranks` is given, asserts the config rank
    matches the allocation (catches rank_pattern key mismatch / global-r collapse)."""
    cov_keys = list(chosen_covs.keys())
    maxerr, n_applied = 0.0, 0
    for name, m in peft_model.named_modules():
        if not (isinstance(m, BaseTunerLayer) and adapter in getattr(m, "lora_A", {})):
            continue
        key = _resolve_covkey(name, cov_keys)
        if key is None:
            continue
        r_l = m.lora_A[adapter].weight.shape[0]                # per-layer rank PEFT resolved from rank_pattern
        if ranks is not None and int(ranks.get(key, r_l)) != int(r_l):
            raise RuntimeError(f"cordapp: layer {name} built at r={r_l} but allocation says "
                               f"r={ranks.get(key)} — rank_pattern key mismatch (use FQ names).")
        base = m.get_base_layer(); W0 = base.weight.data.float().clone()  # clone: correct err even in fp32
        B, A, W_res = kpm_BAR(W0, chosen_covs[key], r_l)
        sc = m.scaling[adapter]
        m.lora_A[adapter].weight.data.copy_(A.to(base.weight.dtype))
        m.lora_B[adapter].weight.data.copy_((B / sc).to(base.weight.dtype))
        base.weight.data.copy_(W_res.to(base.weight.dtype))
        eff = base.weight.data.float() + sc * (m.lora_B[adapter].weight.data.float()
                                               @ m.lora_A[adapter].weight.data.float())
        maxerr = max(maxerr, (eff - W0).abs().max().item())
        n_applied += 1
    if n_applied == 0:
        raise RuntimeError("cordapp: no target LoRA layers matched chosen_covs keys (name mismatch)")
    return maxerr


def finalize_dynamic_rank_config(out_dir):
    """AFTER residual_save.convert_saved_to_w0_relative stacks each layer to rank-2r^l, PEFT's
    adapter_config.json still holds the ORIGINAL per-layer rank_pattern/alpha_pattern (residual_save only
    doubles the global r/alpha). Double them here so a reload builds each layer at 2r^l with scaling
    preserved (alpha^l/r^l = (4r^l)/(2r^l) = 2). No-op for non-dynamic-rank configs. Returns #layers."""
    import os, json
    p = os.path.join(out_dir, "adapter_config.json")
    with open(p) as f:
        cfg = json.load(f)
    rp = cfg.get("rank_pattern") or {}
    ap = cfg.get("alpha_pattern") or {}
    if rp:
        cfg["rank_pattern"] = {k: 2 * int(v) for k, v in rp.items()}
    if ap:
        cfg["alpha_pattern"] = {k: 2 * int(v) for k, v in ap.items()}
    with open(p, "w") as f:
        json.dump(cfg, f, indent=2)
    return len(rp)


# ============================== one-shot precompute (used by the trainer) ==============================
@torch.no_grad()
def precompute_cordapp(raw_model, prompts, tokenizer, targets, fixed_rank,
                       N=DEFAULT_N, calib_size=256, max_len=256, bs=4, scaling=2.0,
                       cov_device="cpu"):
    """End-to-end on the RAW model (BEFORE get_peft_model), STREAMING to bound memory:
      for each of <=N calib rounds -> one covariance per layer -> score it (Eq 8) -> keep the best cov +
      its spectrum per layer -> free the round. Then allocate ranks to tau=fixed_rank*sum(d_in+d_out)
      (Eq 9-10) -> build patterns. Peak cov memory ~ 2 x (single covariance set), not N x.

    Weights/SVDs run on the model's device (fast); the retained best covs are offloaded to `cov_device`
    (default CPU) and moved back to the layer device inside apply_cordapp/kpm_BAR. Selecting per-layer
    argmin streaming is identical to select_covariances over the full pool (argmin is order-independent).

    Returns dict: chosen_covs, ranks, rank_pattern, alpha_pattern, realized_tau, nominal_tau, meta, sel_info.
    Caller: LoraConfig(r=fixed_rank, lora_alpha=int(scaling*fixed_rank), target_modules=targets,
    rank_pattern=rank_pattern, alpha_pattern=alpha_pattern) -> get_peft_model -> apply_cordapp(chosen_covs,
    ranks)."""
    weights = {n: m.weight for n, m in _target_linears(raw_model, targets)}   # live refs (on model device)
    best_cov, best_S, best_pi, sel_info = {}, {}, {}, {}
    for ri, rp in enumerate(_calib_rounds(prompts, N, calib_size)):
        rc = _round_cov(raw_model, rp, tokenizer, targets, max_len, bs)
        for name, C in rc.items():
            if name not in weights:
                continue
            score, S, pi = cov_selection_score(C, weights[name])
            if name not in sel_info or score < sel_info[name]["score"]:
                best_cov[name] = C.to(cov_device)
                best_S[name] = S.to(cov_device)
                best_pi[name] = pi
                sel_info[name] = {"chosen_round": ri, "score": score, "pi": pi}
        del rc
    W_shapes = {n: weights[n] for n in best_cov}
    tau = budget_tau(W_shapes, fixed_rank)
    ranks, realized, meta = allocate_ranks(best_cov, best_S, W_shapes, tau, pis=best_pi)
    rank_pattern, alpha_pattern = build_patterns(ranks, scaling=scaling)
    return {
        "chosen_covs": best_cov, "ranks": ranks,
        "rank_pattern": rank_pattern, "alpha_pattern": alpha_pattern,
        "realized_tau": realized, "nominal_tau": tau,
        "meta": meta, "sel_info": sel_info,
    }
