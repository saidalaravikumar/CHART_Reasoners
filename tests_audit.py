"""tests_audit.py -- executable audit of the CHART mathematics and code (run before experiments).
T1 stability halting on a hand-made example; T2 Theorem 2 (Ville) type-I control of the consensus
e-process; T3 Corollary 1 error decomposition; T4 Lemma 1 + Theorem 1 (LTT) family-wise validity;
T5 Proposition 2 (Wald) expected sample size; T6 Proposition 3 (Hoeffding) width bound."""
import numpy as np
from chart import *
rng = np.random.default_rng(0); ok = True
def check(name, cond, msg=''):
    global ok; ok &= bool(cond); print(('PASS ' if cond else 'FAIL ') + name, msg)

# T1
ans = np.array([[[1, 2, 2, 2, 3, 3, 3, 3]]]); idx = stability_halt(ans, 2)
check('T1 stability halt m=2', idx[0, 0] == 3, f'idx={idx[0,0]} (expected 3: a1=a2=a3=2)')
idx1 = stability_halt(ans, 1); check('T1 stability halt m=1', idx1[0, 0] == 2)
ans2 = np.array([[[1, 2, 3, 4]]]); check('T1 forced halt at Tmax', stability_halt(ans2, 1)[0, 0] == 3)

# T2: all answers have mass <= theta0 -> P(accept) <= beta, for several k0
n, K = 20000, 16
for theta0, beta, k0 in [(0.5, 0.1, 1), (0.5, 0.05, 3), (0.7, 0.1, 2), (0.6, 0.2, 1)]:
    # answer distribution: mass theta0 on value 0, rest spread uniformly over many values
    A = np.where(rng.random((n, K)) < theta0, 0, rng.integers(1, 1000, (n, K)))
    C = np.zeros((n, K), bool); cost = np.ones((n, K))
    acc, _, _, _ = consensus_eprocess(A, C, cost, k0, theta0, beta, K)
    se = np.sqrt(beta*(1-beta)/n)
    check(f'T2 Ville type-I theta0={theta0} beta={beta} k0={k0}', acc.mean() <= beta + 3*se, f'P(accept)={acc.mean():.4f} <= {beta}')

# T3: error decomposition P(err & acc) <= beta + eps(theta0), with mixture of instance types
theta0, beta = 0.5, 0.1; n = 20000
typ = rng.integers(0, 3, n)            # 0: correct dominant (0.8), 1: wrong dominant (0.7), 2: no consensus
pc = np.where(typ == 0, 0.8, np.where(typ == 1, 0.1, 0.3)); pw = np.where(typ == 1, 0.7, np.where(typ == 0, 0.1, 0.3))
u = rng.random((n, K)); A = np.where(u < pc[:, None], 0, np.where(u < (pc+pw)[:, None], 1, rng.integers(2, 1000, (n, K))))
C = (A == 0); acc, cc, _, _ = consensus_eprocess(A, C, np.ones((n, K)), 1, theta0, beta, K)
eps = np.mean(np.where(pw > theta0, pw, 0.0))
check('T3 error decomposition', (acc & ~cc).mean() <= beta + eps + 0.01, f'P(err,acc)={(acc&~cc).mean():.4f} <= beta+eps={beta+eps:.4f}')

# T4: LTT FWER <= delta.  M configs with true selective risks around alpha
alpha, delta, M, ncal, reps = 0.1, 0.1, 30, 1000, 1000
true_r = np.linspace(0.05, 0.2, M); cov = np.linspace(0.9, 0.4, M)
bad = 0; bad_b = 0; bad_naive = 0
for r in range(reps):
    ACC = rng.random((M, ncal)) < cov[:, None]; ERR = rng.random((M, ncal)) < true_r[:, None]
    COST = np.ones((M, ncal)); cal = np.arange(ncal)
    _, cert = ltt_select(ACC, ERR, COST, cal, alpha, delta, method='fst', rng=rng)
    bad += any(true_r[m] > alpha for m in cert)
    _, cert = ltt_select(ACC, ERR, COST, cal, alpha, delta, method='bonf', rng=rng); bad_b += any(true_r[m] > alpha for m in cert)
    _, cert = ltt_select(ACC, ERR, COST, cal, alpha, delta, method='naive', rng=rng); bad_naive += any(true_r[m] > alpha for m in cert)
check('T4 LTT-FST FWER<=delta', bad/reps <= delta + 3*np.sqrt(delta*(1-delta)/reps), f'FWER={bad/reps:.3f}')
check('T4 LTT-Bonferroni FWER<=delta', bad_b/reps <= delta + 3*np.sqrt(delta*(1-delta)/reps), f'FWER={bad_b/reps:.3f}')
print('     (naive empirical selection FWER =', bad_naive/reps, '-- expected to exceed delta)')

# T4b: a-priori ordered FST with skipping of tiny acceptance sets (threshold family)
bad = 0; reps = 1000; ncal = 800; alpha = 0.05
for r in range(reps):
    s = rng.random(ncal); err = rng.random(ncal) < 0.02 + 0.3*(1 - s)**2       # P(err|s) decreasing in s
    th = np.linspace(0.999, 0.0, 60); ACC = np.array([s >= g for g in th]); ERR = np.tile(err, (60, 1))
    true = np.array([np.mean(0.02 + 0.3*(1-np.linspace(g, 1, 2001))**2) for g in th])   # exact selective risk
    _, cert = ltt_select(ACC, ERR, np.ones((60, ncal)), np.arange(ncal), alpha, 0.1, method='fst_given', rng=rng)
    bad += any(true[m] > alpha for m in cert)
check('T4b ordered FST with skip FWER<=delta', bad/reps <= 0.1 + 3*np.sqrt(0.09/reps), f'FWER={bad/reps:.3f}')

# T5: Wald bound with constant Kelly bet
theta0, beta = 0.5, 0.05
for p in (0.7, 0.8, 0.9):
    b = min((p-theta0)/(theta0*(1-theta0)), 0.95/theta0); g = p*np.log1p(b*(1-theta0)) + (1-p)*np.log1p(-b*theta0)
    N = []
    for r in range(4000):
        L = 0; k = 0
        while L < np.log(1/beta): X = rng.random() < p; L += np.log1p(b*(X-theta0)); k += 1
        N.append(k)
    bound = (np.log(1/beta) + np.log1p(b*(1-theta0)))/g
    check(f'T5 Wald E[N] p={p}', np.mean(N) <= bound, f'E[N]={np.mean(N):.2f} <= {bound:.2f}  (log(1/beta)/KL={np.log(1/beta)/g:.2f})')

# T6: Hoeffding width bound for plurality of K draws
for p in (0.6, 0.75, 0.9):
    for K in (4, 8, 16):
        cnt = rng.binomial(K, p, 200000); emp = (cnt <= K/2).mean(); bnd = np.exp(-2*K*(p-0.5)**2)
        check(f'T6 Hoeffding p={p} K={K}', emp <= bnd + 1e-3, f'P(count<=K/2)={emp:.4f} <= {bnd:.4f}')
# ---------------- version-3 additions ----------------
# T7: mixture e-process type-I control (Theorem 2 for the mixture bet)
n, K = 20000, 16
for theta0, beta, k0 in [(0.5, 0.1, 1), (0.5, 0.05, 3), (0.7, 0.1, 2), (0.3, 0.2, 1)]:
    A = np.where(rng.random((n, K)) < theta0, 0, rng.integers(1, 1000, (n, K)))
    acc, _, _, _ = consensus_mixture(A, np.zeros((n, K), bool), np.ones((n, K)), k0, theta0, beta, K)
    check(f'T7 mixture Ville type-I theta0={theta0} beta={beta} k0={k0}', acc.mean() <= beta + 3*np.sqrt(beta*(1-beta)/n), f'P(accept)={acc.mean():.4f}')
# T8: mixture budget bound E[N] <= min_g (log(G/beta) + log(1+b_g(1-theta0)))/g(b_g), no truncation
theta0, beta, G = 0.5, 0.1, 8; bmax = 0.95/theta0; bs = bmax*np.geomspace(1/16, 1, G)
for p in (0.7, 0.85, 0.95, 1.0):
    Kbig = 400; X = rng.random((3000, Kbig)) < p
    logW = np.zeros((3000, G)); N = np.full(3000, Kbig)
    for j in range(Kbig):
        logW += np.log1p(bs[None, :]*(X[:, j:j+1] - theta0)); m = logW.max(1); logE = m + np.log(np.exp(logW - m[:, None]).mean(1))
        hit = (logE >= np.log(1/beta)) & (N == Kbig); N[hit] = j + 1
    gs = p*np.log1p(bs*(1-theta0)) + (1-p)*np.log1p(-bs*theta0)
    bnd = np.min(np.where(gs > 0, (np.log(G/beta) + np.log1p(bs*(1-theta0)))/np.where(gs > 0, gs, 1), np.inf))
    check(f'T8 mixture budget bound p={p}', N.mean() <= bnd, f'E[N]={N.mean():.2f} <= {bnd:.2f}')
# T9: a-priori ordered FST: plain, D1-skip (valid) and D2-skip on a family built to stress the skip rule
reps = 2000; alpha, delta = 0.05, 0.1; ncal = 600; bad = {'given_plain': 0, 'given_d1skip': 0, 'given_d2skip': 0}
M = 40; nmin = int(np.ceil(np.log(delta)/np.log(1-alpha)))
cov = np.r_[np.full(20, nmin*0.9/ncal), np.linspace(0.3, 0.9, 20)]   # many barely-testable configurations first
tr = np.r_[np.full(20, 0.08), np.linspace(0.01, 0.12, 20)]
for r in range(reps):
    ACC = rng.random((M, ncal)) < cov[:, None]; ERR = rng.random((M, ncal)) < tr[:, None]
    for meth in bad:
        _, cert = ltt_select_v3(ACC, ERR, np.ones((M, ncal)), np.arange(ncal), alpha, delta, method=meth, rng=rng)
        bad[meth] += any(tr[m] > alpha for m in cert)
for meth, b in bad.items():
    print(f'     FWER {meth}: {b/reps:.3f}')
check('T9 D1-skip FWER<=delta', bad['given_d1skip']/reps <= delta + 3*np.sqrt(delta*(1-delta)/reps))
check('T9 plain FWER<=delta', bad['given_plain']/reps <= delta + 3*np.sqrt(delta*(1-delta)/reps))
print('ALL PASS' if ok else 'SOME FAILED')
