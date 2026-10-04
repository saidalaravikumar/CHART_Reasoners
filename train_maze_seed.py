"""train_maze_seed.py -- retrains the maze reasoner with a different random seed (initialization, batch order and
PTRM noise) for the seed-robustness study.  Same architecture, loss, deep-supervision curriculum, noise schedule,
optimizer and number of optimizer steps (31000) as the seed-0 run of train.py; the curriculum and learning-rate
schedule are indexed by step so that runs on machines of different speed are identical in expectation.
Resumable in wall-clock chunks.   usage: SEED=1 python train_maze_seed.py CHUNK_SECONDS"""
import os, sys, time, copy, json, numpy as np, torch, torch.nn.functional as F
from common import *
torch.set_num_threads(int(os.environ.get('THREADS', 4)))
SEED = int(os.environ.get('SEED', 1)); chunk = float(sys.argv[1]); TOTAL = 31000
CFG = dict(C=32, n=4, T_cyc=1, N_sup_min=2, N_sup_max=6, bs=64, ntrain=40000, lr=3e-3, seed=SEED, steps=TOTAL)
if not os.path.exists('maze_train.npz'):
    X, Y, _ = maze_dataset(40000, 15, seed=1); Xv, Yv, _ = maze_dataset(500, 15, seed=999)
    np.savez_compressed('maze_train.npz', X=X.astype(np.uint8), Y=Y, Xv=Xv.astype(np.uint8), Yv=Yv)
d = np.load('maze_train.npz'); X, Y = torch.tensor(d['X']).float(), torch.tensor(d['Y']); Xv, Yv = torch.tensor(d['Xv']).float(), torch.tensor(d['Yv'])
tag = f'maze_s{SEED}'
model = TinyRecursiveReasoner('maze', CFG['C'], CFG['n'], CFG['T_cyc'])
if os.path.exists(f'state_{tag}.pt'):
    S = torch.load(f'state_{tag}.pt', weights_only=False); model.load_state_dict(S['model']); ema = copy.deepcopy(model); ema.load_state_dict(S['ema'])
    opt = torch.optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=0.01, betas=(0.9, 0.95)); opt.load_state_dict(S['opt'])
    st = S['st']; torch.set_rng_state(S['rng'])
else:
    torch.manual_seed(1000 + SEED); model = TinyRecursiveReasoner('maze', CFG['C'], CFG['n'], CFG['T_cyc']); ema = copy.deepcopy(model)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=0.01, betas=(0.9, 0.95)); st = dict(step=0, log=[])

def task_loss(logits, y, x):
    m = (x[:, 0] < 0.5).float()
    l = F.binary_cross_entropy_with_logits(logits, y.float(), reduction='none', pos_weight=torch.tensor(3.0))
    return (l*m).sum()/m.sum()

@torch.no_grad()
def evaluate(net, T=12):
    net.eval(); xe = net.embed(Xv); y, z = net.init_state(xe, None); acc = []
    for t in range(T):
        y, z = net.outer_step(xe, y, z, grad=False); acc.append(exact_correct('maze', predict('maze', net.decode(y), Xv), Yv).float().mean().item())
    net.train(); return acc

t0 = time.time()
while time.time() - t0 < chunk and st['step'] < TOTAL:
    idx = torch.randint(0, len(X), (CFG['bs'],)); x, yl = X[idx], Y[idx]; sig = torch.rand(len(idx))
    frac0 = st['step']/TOTAL; N_sup = CFG['N_sup_min'] + int(round((CFG['N_sup_max']-CFG['N_sup_min'])*frac0))
    with torch.no_grad(): y, z = model.init_state(model.embed(x), sig)
    for s in range(N_sup):
        frac = min(1.0, st['step']/TOTAL)
        for g in opt.param_groups: g['lr'] = CFG['lr']*min(1.0, st['step']/300)*(0.5*(1+np.cos(np.pi*frac))*0.95+0.05)
        xe = model.embed(x); y, z = model.outer_step(xe, y, z, grad=True, sigma=sig)
        logits = model.decode(y); q = model.qvalue(y)
        with torch.no_grad(): corr = exact_correct('maze', predict('maze', logits, x), yl).float()
        loss = task_loss(logits, yl, x) + 0.5*F.binary_cross_entropy_with_logits(q, corr)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); st['step'] += 1
        with torch.no_grad():
            dd = min(0.999, (1+st['step'])/(10+st['step']))
            for pe, pm in zip(ema.parameters(), model.parameters()): pe.mul_(dd).add_(pm, alpha=1-dd)
        y, z = y.detach(), z.detach()
acc = evaluate(ema); st['log'].append(dict(step=st['step'], acc=acc))
print(f"seed {SEED} step {st['step']}/{TOTAL} loss {loss.item():.3f} val-acc(ema) {[round(a,3) for a in acc][::2]}", flush=True)
torch.save(dict(model=model.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(), st=st, rng=torch.get_rng_state()), f'state_{tag}.pt')
torch.save(ema.state_dict(), f'ckpt_{tag}.pt'); json.dump(dict(cfg=CFG, log=st['log']), open(f'trainlog_{tag}.json', 'w'))
