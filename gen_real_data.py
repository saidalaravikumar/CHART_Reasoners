"""gen_real_data.py -- builds every practical instance set once (so both machines use identical data).
Training/validation windows come from the 6 training cities; calibration/test pools from the 4 unseen
cities (a city-level split); the indoor shift pool from the Dragon Age: Origins maps."""
import numpy as np
from realmaps import *
G = 15
sets = dict(street_train=(40000, 11, 'street', TRAIN_CITIES), street_val=(500, 12, 'street', TRAIN_CITIES),
            street_main=(3000, 21, 'street', TEST_CITIES), street_fresh=(6000, 22, 'street', TEST_CITIES),
            dao_main=(2000, 31, 'dao', None))
for name, (n, seed, dom, cities) in sets.items():
    kw = dict(cities=cities, f=2) if dom == 'street' else {}
    X, Y, L, Mi = real_dataset(n, G, seed, dom, **kw)
    np.savez_compressed(f'{name}.npz', X=X.astype(np.uint8), Y=Y, D=L, M=Mi)
    print(name, X.shape, 'mean L', L.mean(), 'corridor cells', Y.reshape(n, -1).sum(1).mean(), flush=True)
