import numpy as np
import cupy
from sklearn.utils import murmurhash3_32 as skhash
from mmh3 import murmurhash3_32 as myhash


n_seeds, n_keys, key_length = 1, 1, 4

keys = np.random.randint(low=0, high=2**32, size=(n_seeds, n_keys, key_length), dtype=np.uint32)
seeds = np.random.randint(low=0, high=2**32, size=(n_seeds, ), dtype=np.uint32)
keys_cp, seeds_cp = cupy.array(keys), cupy.array(seeds)

numpy_results = myhash(keys, seeds)
cupy_results = myhash(keys_cp, seeds_cp).get()

for i in range(n_seeds):
    for j in range(n_keys):
        sk_result: int = skhash(
            key=keys[i, j].tobytes(),
            seed=int(seeds[i]),
            positive=True
        )
        assert sk_result == int(numpy_results[i, j])
        assert sk_result == int(cupy_results[i, j])
