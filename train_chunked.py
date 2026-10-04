"""train_chunked.py -- identical recipe to train.py, but resumable in wall-clock chunks
(used to train the Sudoku reasoner on a second machine inside short sessions).
usage: python train_chunked.py sudoku TOTAL_MINUTES CHUNK_SECONDS"""
import os, sys, time, copy, json, numpy as np, torch, torch.nn.functional as F
from common import *
torch.set_num_threads(int(os.environ.get('THREADS', 4)))
task = sys.argv[1]; minutes = float(sys.argv[2]); chunk = float(sys.argv[3])
CFG = dict(C=64, n=4, T_cyc=1, N_sup_min=2, N_sup_max=6, bs=64, lr=2e-3)
SIG_MAX = 1.0
d = np.load('sudoku_train.npz'); X, Y = torch.tensor(d['P']).long(), torch.tensor(d['S']).long()
d = np.load('sudoku_val.npz'); Xv, Yv = torch.tensor(d['P']).long(), torch.tensor(d['S']).long()
model = TinyRecursiveReasoner(task, CFG['C'], CFG['n'], CFG['T_cyc']); ema = copy.deepcopy(model)
opt = torch.optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=0.01, betas=(0.9, 0.95))
st = dict(step=0, trained=0.0, ep=0, log=[])
if os.path.exists(f'state_{task}.pt'):
    S = torch.load(f'state_{task}.pt', weights_only=False)
    model.load_state_dict(S['model']); ema.load_state_dict(S['ema']); opt.load_state_dict(S['opt']); st = S['st']
    torch.set_rng_state(S['rng'])
else:
    torch.manual_seed(0)

def task_loss(logits, y, x):
    m = (x == 0).float()
    ce = F.cross_entropy(logits.reshape(-1, 9), (y - 1).reshape(-1), reduction='none').view_as(m)
    return (ce*m).sum()/m.sum()

@torch.no_grad()
def evaluate(net, T=12):
    net.eval(); xe = net.embed(Xv); y, z = net.init_state(xe, None); acc = []
    for t in range(T):
        y, z = net.outer_step(xe, y, z, grad=False)
        acc.append(exact_correct(task, predict(task, net.decode(y), Xv), Yv).float().mean().item())
    net.train(); return acc

t_start = time.time(); total = minutes*60
while time.time() - t_start < chunk and st['trained'] < total:
    t_it = time.time()
    idx = torch.randint(0, len(X), (CFG['bs'],)); x, yl = X[idx], Y[idx]
    sig = SIG_MAX*torch.rand(len(idx))
    frac0 = min(1.0, st['trained']/total)
    N_sup = CFG['N_sup_min'] + int(round((CFG['N_sup_max']-CFG['N_sup_min'])*frac0))
    with torch.no_grad(): y, z = model.init_state(model.embed(x), sig)
    for s in range(N_sup):
        frac = min(1.0, st['trained']/total)
        for g in opt.param_groups: g['lr'] = CFG['lr']*min(1.0, st['step']/300)*(0.5*(1+np.cos(np.pi*frac))*0.95+0.05)
        xe = model.embed(x); y, z = model.outer_step(xe, y, z, grad=True, sigma=sig)
        logits = model.decode(y); q = model.qvalue(y)
        with torch.no_grad(): corr = exact_correct(task, predict(task, logits, x), yl).float()
        loss = task_loss(logits, yl, x) + 0.5*F.binary_cross_entropy_with_logits(q, corr)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); st['step'] += 1
        with torch.no_grad():
            dd = min(0.999, (1+st['step'])/(10+st['step']))
            for pe, pm in zip(ema.parameters(), model.parameters()): pe.mul_(dd).add_(pm, alpha=1-dd)
        y, z = y.detach(), z.detach()
    st['trained'] += time.time() - t_it
acc = evaluate(ema); st['log'].append(dict(step=st['step'], trained=st['trained'], acc=acc))
print(f"step {st['step']} trained {st['trained']/60:.1f} min  loss {loss.item():.3f}  val-acc(ema) {[round(a,3) for a in acc][::2]}", flush=True)
torch.save(dict(model=model.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(), st=st, rng=torch.get_rng_state()), f'state_{task}.pt')
torch.save(ema.state_dict(), f'ckpt_{task}.pt')
json.dump(dict(cfg=CFG, log=st['log']), open(f'trainlog_{task}.json', 'w'))
