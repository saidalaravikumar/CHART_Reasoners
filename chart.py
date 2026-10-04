"""
chart.py -- statistical layer of CHART (Certified Halting And Risk-controlled Trajectories).
Operates on recorded trajectory pools:  ans[n,K,T] (answer hashes), cor[n,K,T] (exact correctness),
q[n,K,T] (Q-head logits).  Trajectories k are i.i.d. given the instance (independent PTRM noise).
"""
import numpy as np
from scipy.stats import binom

# ---------------------------------------------------------------- depth: stability halting
def stability_halt(ans, m, Tmax=None):
    """Per-trajectory halting index: first t>=m with ans[t]==ans[t-1]==...==ans[t-m]
    (answer unchanged over m consecutive outer steps); otherwise Tmax-1.
    Returns idx[n,K] (0-based step index); cost = idx+1 outer steps."""
    n, K, T = ans.shape; Tm = T if Tmax is None else Tmax
    same = np.ones((n, K, Tm), bool); same[:, :, :m] = False
    for j in range(1, m+1):
        same[:, :, m:] &= (ans[:, :, m:Tm] == ans[:, :, m-j:Tm-j])
    same[:, :, Tm-1] = True
    return same.argmax(2)

def halted_outputs(ans, cor, idx):
    ii = np.arange(ans.shape[0])[:, None]; kk = np.arange(ans.shape[1])[None, :]
    return ans[ii, kk, idx], cor[ii, kk, idx], idx + 1

# ---------------------------------------------------------------- width: consensus e-process
def consensus_eprocess(A, C, cost, k0, theta0, beta, Kmax, cfrac=0.95, futility=True):
    """Anytime-valid consensus certificate (Theorem 2).
    A,C,cost: [n,K] halted answers / correctness / cost of i.i.d. trajectories.
    Warm-up: first k0 trajectories -> anchor v = plurality (ties: earliest).
    Test: fresh draws X_j = 1{A_j == v};  E_k = prod_j (1 + b_j (X_j - theta0)),
    predictable Kelly plug-in b_j in [0, cfrac/theta0].  Accept v when E_k >= 1/beta.
    Returns accept[n], correct[n] (of anchor), used trajectories[n], compute[n] (sum of costs)."""
    n = A.shape[0]
    anchor = np.empty(n, A.dtype); anc_cor = np.zeros(n, bool); agree0 = np.zeros(n)
    for i in range(n):
        w = A[i, :k0]; vals, first, cnt = np.unique(w, return_index=True, return_counts=True)
        best = np.flatnonzero(cnt == cnt.max()); j = best[np.argmin(first[best])]
        anchor[i] = vals[j]; anc_cor[i] = C[i, first[j]]; agree0[i] = cnt[j]
    logE = np.zeros(n); S = np.zeros(n); done = np.zeros(n, bool); acc = np.zeros(n, bool)
    used = np.full(n, k0); comp = cost[:, :k0].sum(1).astype(float)
    if beta >= 1.0:                                          # degenerate certificate: accept the warm-up plurality
        return np.ones(n, bool), anc_cor, used, comp
    bmax = cfrac/theta0; thr = np.log(1/beta); up = np.log1p(bmax*(1-theta0))
    for j in range(k0, Kmax):
        act = ~done
        if not act.any(): break
        k = j - k0                                           # number of past test draws
        phat = (S + agree0 + 0.5)/(k + k0 + 1.0)             # predictable estimate
        b = np.clip((phat - theta0)/(theta0*(1-theta0)), 0, bmax)
        X = (A[:, j] == anchor).astype(float)
        logE = np.where(act, logE + np.log1p(b*(X - theta0)), logE)
        S = np.where(act, S + X, S)
        comp = np.where(act, comp + cost[:, j], comp); used = np.where(act, used + 1, used)
        newacc = act & (logE >= thr - 1e-12)
        acc |= newacc; done |= newacc
        if futility:                                         # cannot reach 1/beta any more -> abstain now
            rem = Kmax - j - 1
            done |= act & ~newacc & (logE + rem*up < thr)
    return acc, anc_cor, used, comp

# ---------------------------------------------------------------- static majority (width K, depth T)
def majority_static(ans, cor, K, t):
    """Plurality over the first K trajectories at step index t. Returns agreement, correct, cost."""
    a = ans[:, :K, t]; c = cor[:, :K, t]; n = a.shape[0]
    agr = np.zeros(n); cc = np.zeros(n, bool)
    for i in range(n):
        vals, first, cnt = np.unique(a[i], return_index=True, return_counts=True)
        best = np.flatnonzero(cnt == cnt.max()); j = best[np.argmin(first[best])]
        agr[i] = cnt[j]/K; cc[i] = c[i, first[j]]
    return agr, cc, np.full(n, K*(t+1.0))

# ---------------------------------------------------------------- Learn-then-Test
def pval_selective(acc, err, alpha):
    """Valid p-value for H: P(err | accept) > alpha  (Lemma 1): P(Bin(n_acc, alpha) <= n_err)."""
    na = acc.sum(); ne = (acc & err).sum()
    return 1.0 if na == 0 else float(binom.cdf(ne, na, alpha))

def ltt_select(ACC, ERR, COST, cal, alpha, delta, mu=0.25, cost_ref=1.0, method='fst', frac1=0.3, rng=None):
    """ACC,ERR: [M,n] bool per config; COST: [M,n]. cal: calibration indices.
    method: 'fst'  -> fixed-sequence testing ordered by p-values on D1 (split of cal), tested on D2
            'bonf' -> Bonferroni over all M configs on the full cal set
            'naive'-> empirical selective risk <= alpha on cal (no guarantee)
    Returns chosen config index (or -1 = abstain on everything) and the certified set."""
    M = ACC.shape[0]
    if method == 'fst':
        perm = rng.permutation(cal); n1 = int(frac1*len(cal)); D1, D2 = perm[:n1], perm[n1:]
        p1 = np.array([pval_selective(ACC[m, D1], ERR[m, D1], alpha) for m in range(M)])
        order = np.lexsort((COST[:, D1].mean(1), p1))       # most promising first
        cert = []
        for m in order:
            if pval_selective(ACC[m, D2], ERR[m, D2], alpha) <= delta: cert.append(m)
            else: break
    elif method == 'fst_given':                             # a-priori order (e.g. threshold strict -> loose)
        # configurations accepting fewer than n_min calibration points can never be rejected
        # ((1-alpha)^n > delta); they are skipped. The skip depends only on acceptance indicators,
        # conditional on which the p-values remain valid, so FWER control is preserved.
        n_min = int(np.ceil(np.log(delta)/np.log(1 - alpha)))
        cert = []
        for m in range(M):
            if ACC[m, cal].sum() < n_min: continue
            if pval_selective(ACC[m, cal], ERR[m, cal], alpha) <= delta: cert.append(m)
            else: break
    elif method == 'bonf':
        cert = [m for m in range(M) if pval_selective(ACC[m, cal], ERR[m, cal], alpha) <= delta/M]
    else:
        cert = []
        for m in range(M):
            na = ACC[m, cal].sum()
            if na > 0 and (ACC[m, cal] & ERR[m, cal]).sum()/na <= alpha: cert.append(m)
    if not cert: return -1, cert
    cert = np.array(cert)
    J = (1 - ACC[cert][:, cal].mean(1)) + mu*COST[cert][:, cal].mean(1)/cost_ref
    return int(cert[np.argmin(J)]), cert

def evaluate(ACC, ERR, COST, m, test):
    if m < 0: return dict(risk=0.0, cov=0.0, cost=0.0, nacc=0)
    a = ACC[m, test]; e = ERR[m, test] & a
    return dict(risk=float(e.sum()/max(a.sum(), 1)), cov=float(a.mean()), cost=float(COST[m, test].mean()), nacc=int(a.sum()))


# =====================================================================================
# Mixture bet with a provable budget, alternative stopping
# rules for ablation, D1-based skipping for a-priori ordered LTT, wealth paths.
# =====================================================================================
def _anchor(A, C, k0):
    n = A.shape[0]; anchor = np.empty(n, A.dtype); anc_cor = np.zeros(n, bool); agree0 = np.zeros(n)
    for i in range(n):
        vals, first, cnt = np.unique(A[i, :k0], return_index=True, return_counts=True)
        best = np.flatnonzero(cnt == cnt.max()); j = best[np.argmin(first[best])]
        anchor[i] = vals[j]; anc_cor[i] = C[i, first[j]]; agree0[i] = cnt[j]
    return anchor, anc_cor, agree0

MIX_GRID = None
def consensus_mixture(A, C, cost, k0, theta0, beta, Kmax, cfrac=0.95, G=8, futility=True, return_paths=False):
    """Mixture-of-constant-bets e-process: E_j = (1/G) sum_g prod_i (1 + b_g (X_i - theta0)),
    b_g on a geometric grid in (0, cfrac/theta0].  An average of nonnegative supermartingales is a
    nonnegative supermartingale (Theorem 2 applies unchanged) and E_j >= E_j^{(g)}/G for every g,
    which gives the budget bound of Proposition 1 for the procedure actually run."""
    n = A.shape[0]
    anchor, anc_cor, _ = _anchor(A, C, k0)
    used = np.full(n, k0); comp = cost[:, :k0].sum(1).astype(float)
    if beta >= 1.0: return np.ones(n, bool), anc_cor, used, comp
    bmax = cfrac/theta0; bs = bmax*np.geomspace(1/16, 1, G)
    logW = np.zeros((n, G)); done = np.zeros(n, bool); acc = np.zeros(n, bool)
    thr = np.log(1/beta); up = np.log1p(bmax*(1-theta0)); paths = []
    for j in range(k0, Kmax):
        act = ~done
        if not act.any(): break
        X = (A[:, j] == anchor).astype(float)[:, None]
        logW = np.where(act[:, None], logW + np.log1p(bs[None, :]*(X - theta0)), logW)
        m = logW.max(1); logE = m + np.log(np.exp(logW - m[:, None]).mean(1))
        comp = np.where(act, comp + cost[:, j], comp); used = np.where(act, used + 1, used)
        if return_paths: paths.append(np.where(act, logE, np.nan))
        newacc = act & (logE >= thr - 1e-12); acc |= newacc; done |= newacc
        if futility: done |= act & ~newacc & (logE + (Kmax - j - 1)*up < thr)
    if return_paths: return acc, anc_cor, used, comp, np.array(paths).T
    return acc, anc_cor, used, comp

def kelly_paths(A, C, cost, k0, theta0, beta, Kmax, cfrac=0.95):
    """Same as consensus_eprocess (plug-in bet) but also returns log-wealth paths (NaN after stopping)."""
    n = A.shape[0]; anchor, anc_cor, agree0 = _anchor(A, C, k0)
    logE = np.zeros(n); S = np.zeros(n); done = np.zeros(n, bool); acc = np.zeros(n, bool)
    bmax = cfrac/theta0; thr = np.log(1/beta); up = np.log1p(bmax*(1-theta0)); P = []; status = np.zeros(n, int)
    for j in range(k0, Kmax):
        act = ~done; k = j - k0
        phat = (S + agree0 + 0.5)/(k + k0 + 1.0); b = np.clip((phat - theta0)/(theta0*(1-theta0)), 0, bmax)
        X = (A[:, j] == anchor).astype(float)
        logE = np.where(act, logE + np.log1p(b*(X - theta0)), logE); S = np.where(act, S + X, S)
        P.append(np.where(act, logE, np.nan))
        newacc = act & (logE >= thr - 1e-12); acc |= newacc; done |= newacc; status[newacc] = 1
        fut = act & ~newacc & (logE + (Kmax - j - 1)*up < thr); done |= fut; status[fut] = 2
    return np.array(P).T, anc_cor, acc, status

# ---------------------------------------------------------------- heuristic stopping rules (E2)
def stop_ragree(A, C, cost, k0, r, s, Kmax):
    """Accept the warm-up anchor once r fresh trajectories agree with it; abstain after s disagreements."""
    n = A.shape[0]; anchor, anc_cor, _ = _anchor(A, C, k0)
    ag = np.zeros(n); dis = np.zeros(n); done = np.zeros(n, bool); acc = np.zeros(n, bool)
    comp = cost[:, :k0].sum(1).astype(float)
    for j in range(k0, Kmax):
        act = ~done
        if not act.any(): break
        X = (A[:, j] == anchor); ag += act & X; dis += act & ~X; comp = np.where(act, comp + cost[:, j], comp)
        newacc = act & (ag >= r); acc |= newacc; done |= newacc | (act & (dis >= s))
    return acc, anc_cor, comp

def stop_beta(A, C, cost, conf, Kmax, kmin=2):
    """Adaptive-Consistency-style Beta stopping: after each trajectory, with a = count of the running
    plurality answer and b = all other answers, stop when P(p > 1/2 | Beta(a+1, b+1)) >= conf and
    accept the plurality; abstain if the budget ends first."""
    from scipy.stats import beta as Bd
    n = A.shape[0]; acc = np.zeros(n, bool); cc = np.zeros(n, bool); comp = np.zeros(n); done = np.zeros(n, bool)
    for i in range(n):
        for k in range(1, Kmax + 1):
            vals, first, cnt = np.unique(A[i, :k], return_index=True, return_counts=True)
            jb = np.argmax(cnt); a = cnt[jb]; b = k - a
            if k >= kmin and Bd.sf(0.5, a + 1, b + 1) >= conf:
                acc[i] = True; cc[i] = C[i, first[jb]]; comp[i] = cost[i, :k].sum(); break
        else:
            comp[i] = cost[i, :Kmax].sum(); cc[i] = C[i, first[jb]]
    return acc, cc, comp

def stop_sprt(A, C, cost, k0, theta0, theta1, Aup, Blo, Kmax):
    """Wald SPRT of p = theta0 against p = theta1 on the agreement indicators with the warm-up anchor.
    Its likelihood ratio is the constant-bet wealth with b = (theta1-theta0)/(theta0(1-theta0)), so the
    acceptance side is also Ville-valid; the lower boundary Blo adds early abstention."""
    n = A.shape[0]; anchor, anc_cor, _ = _anchor(A, C, k0)
    L = np.zeros(n); done = np.zeros(n, bool); acc = np.zeros(n, bool); comp = cost[:, :k0].sum(1).astype(float)
    l1 = np.log(theta1/theta0); l0 = np.log((1-theta1)/(1-theta0))
    for j in range(k0, Kmax):
        act = ~done
        if not act.any(): break
        X = (A[:, j] == anchor); L = np.where(act, L + np.where(X, l1, l0), L); comp = np.where(act, comp + cost[:, j], comp)
        newacc = act & (L >= Aup); acc |= newacc; done |= newacc | (act & (L <= Blo))
    return acc, anc_cor, comp

def majority_halted(A, C, cost, K):
    """Plurality over the first K halted trajectories and its agreement fraction."""
    n = A.shape[0]; agr = np.zeros(n); cc = np.zeros(n, bool)
    for i in range(n):
        vals, first, cnt = np.unique(A[i, :K], return_index=True, return_counts=True)
        best = np.flatnonzero(cnt == cnt.max()); j = best[np.argmin(first[best])]
        agr[i] = cnt[j]/K; cc[i] = C[i, first[j]]
    return agr, cc, cost[:, :K].sum(1).astype(float)

def ltt_select_v3(ACC, ERR, COST, cal, alpha, delta, mu=0.25, cost_ref=1.0, method='fst', frac1=0.3, rng=None):
    """Adds a-priori ordered fixed-sequence testing with three skip policies (E5):
    'given_plain'  : no skipping (stops at the first untestable configuration);
    'given_d2skip' : skip if n_acc(D_cal) < n_min  (version-2 Remark 1; not covered by Theorem 1);
    'given_d1skip' : split D_cal into D1/D2, skip if the D1-predicted D2 count < n_min, test on D2 only.
    The D1 decision is independent of D2, so Theorem 1 applies unchanged."""
    if method in ('fst', 'bonf', 'naive'):
        return ltt_select(ACC, ERR, COST, cal, alpha, delta, mu, cost_ref, method=method, frac1=frac1, rng=rng)
    M = ACC.shape[0]; n_min = int(np.ceil(np.log(delta)/np.log(1 - alpha))); cert = []
    if method == 'given_d1skip':
        perm = rng.permutation(cal); n1 = int(frac1*len(cal)); D1, D2 = perm[:n1], perm[n1:]
        for m in range(M):
            if ACC[m, D1].sum()*len(D2)/max(len(D1), 1) < n_min: continue
            if pval_selective(ACC[m, D2], ERR[m, D2], alpha) <= delta: cert.append(m)
            else: break
    else:
        for m in range(M):
            if method == 'given_d2skip' and ACC[m, cal].sum() < n_min: continue
            if pval_selective(ACC[m, cal], ERR[m, cal], alpha) <= delta: cert.append(m)
            else: break
    if not cert: return -1, cert
    cert = np.array(cert)
    J = (1 - ACC[cert][:, cal].mean(1)) + mu*COST[cert][:, cal].mean(1)/cost_ref
    return int(cert[np.argmin(J)]), cert

def binom_ci(k, n, level=0.95):
    """Clopper-Pearson interval for a binomial proportion (used for violation frequencies)."""
    from scipy.stats import beta as Bd
    lo = 0.0 if k == 0 else Bd.ppf((1-level)/2, k, n-k+1)
    hi = 1.0 if k == n else Bd.ppf(1-(1-level)/2, k+1, n-k)
    return lo, hi
