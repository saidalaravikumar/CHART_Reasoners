"""train.py -- deep-supervision training of the tiny recursive reasoner (TRM recipe + PTRM noise).
usage: python train.py maze|sudoku [minutes]"""
import os, sys, time, copy, json, numpy as np, torch, torch.nn.functional as F
from common import *
torch.set_num_threads(2)
task = sys.argv[1]; minutes = float(sys.argv[2]) if len(sys.argv) > 2 else 60
torch.manual_seed(0); np.random.seed(0)
CFG = dict(maze=dict(C=32, n=4, T_cyc=1, N_sup_min=2, N_sup_max=6, bs=64, ntrain=40000, lr=3e-3),
           sudoku=dict(C=64, n=4, T_cyc=1, N_sup_min=2, N_sup_max=6, bs=64, ntrain=40000, lr=2e-3))[task]
SIG_MAX = 1.0
if task == 'maze':
    GS = int(os.environ.get('GS', 15)); X, Y, _ = maze_dataset(CFG['ntrain'], GS, seed=1); Xv, Yv, _ = maze_dataset(500, GS, seed=999)
    X, Y, Xv, Yv = map(torch.tensor, (X, Y, Xv, Yv))
else:
    d = np.load('sudoku_train.npz'); X, Y = d['P'], d['S']; d = np.load('sudoku_val.npz'); Xv, Yv = d['P'], d['S']
    X, Y, Xv, Yv = map(torch.tensor, (X, Y, Xv, Yv))
print('data ready', X.shape, flush=True)
model = TinyRecursiveReasoner(task, CFG['C'], CFG['n'], CFG['T_cyc'])
ema = copy.deepcopy(model)
print('params', sum(p.numel() for p in model.parameters()), flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=0.01, betas=(0.9, 0.95))

def task_loss(logits, y, x):
    if task == 'maze':
        m = (x[:, 0] < 0.5).float()                      # loss on free cells only
        l = F.binary_cross_entropy_with_logits(logits, y.float(), reduction='none', pos_weight=torch.tensor(3.0))
        return (l*m).sum()/m.sum()
    m = (x == 0).float()                                  # loss on blanks only
    ce = F.cross_entropy(logits.reshape(-1, 9), (y - 1).reshape(-1), reduction='none').view_as(m)
    return (ce*m).sum()/m.sum()

@torch.no_grad()
def evaluate(net, T=12):  # deterministic (sigma=0) exact accuracy per outer step
    net.eval(); xe = net.embed(Xv); y, z = net.init_state(xe, 0.0); acc = []
    for t in range(T):
        y, z = net.outer_step(xe, y, z, grad=False)
        acc.append(exact_correct(task, predict(task, net.decode(y), Xv), Yv).float().mean().item())
    net.train(); return acc

t0 = time.time(); step = 0; ep = 0; total_steps_est = None; log = []
while time.time() - t0 < minutes*60:
    perm = torch.randperm(len(X)); ep += 1
    for b in range(0, len(X) - CFG['bs'] + 1, CFG['bs']):
        idx = perm[b:b+CFG['bs']]; x, yl = X[idx], Y[idx]
        sig = SIG_MAX*torch.rand(len(idx))
        with torch.no_grad():
            y, z = model.init_state(model.embed(x), sig)
        frac0 = min(1.0, (time.time()-t0)/(minutes*60))
        N_sup = CFG['N_sup_min'] + int(round((CFG['N_sup_max']-CFG['N_sup_min'])*frac0))   # deep-supervision curriculum
        for s in range(N_sup):
            frac = min(1.0, (time.time()-t0)/(minutes*60))
            for g in opt.param_groups: g['lr'] = CFG['lr']*min(1.0, step/300)*(0.5*(1+np.cos(np.pi*frac))*0.95+0.05)
            xe = model.embed(x)
            y, z = model.outer_step(xe, y, z, grad=True, sigma=sig)
            logits = model.decode(y); q = model.qvalue(y)
            with torch.no_grad(): corr = exact_correct(task, predict(task, logits, x), yl).float()
            loss = task_loss(logits, yl, x) + 0.5*F.binary_cross_entropy_with_logits(q, corr)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); step += 1
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()): d = min(0.999, (1+step)/(10+step)); pe.mul_(d).add_(pm, alpha=1-d)
            y, z = y.detach(), z.detach()
        if (b//CFG['bs']) % 400 == 0 and b > 0:
            print('  quick-eval(raw)', [round(a,3) for a in evaluate(model)][::2], flush=True)
        if (b//CFG['bs']) % 100 == 0:
            print(f'ep {ep} b {b} step {step} loss {loss.item():.4f} acc_last {corr.mean().item():.3f} t {time.time()-t0:.0f}s', flush=True)
        if time.time() - t0 > minutes*60: break
    acc = evaluate(ema); log.append(dict(ep=ep, step=step, acc=acc, t=time.time()-t0)); print('EVAL ema', ep, [round(a, 3) for a in acc], flush=True)
    torch.save(ema.state_dict(), f'ckpt_{task}.pt'); json.dump(dict(cfg=CFG, log=log), open(f'trainlog_{task}.json', 'w'))
print('done', time.time()-t0)
