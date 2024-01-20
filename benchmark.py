from time import perf_counter
from itertools import product

import numpy as np
import cupy
import matplotlib.pyplot as plt
from sklearn.utils import murmurhash3_32 as sk_mmh3
from mmh3 import murmurhash3_32 as my_mmh3


if __name__ == "__main__":
    n_keys_list = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    n_seeds_list = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    key_length = 4

    sklearn_times = []
    my_numpy_times = []
    my_cupy_times = []
    warmup = False

    for n_keys, n_seeds in product(n_keys_list, n_seeds_list):
        print(f"n_keys(bytes): {n_keys} n_seeds: {n_seeds}")
        keys = np.random.randint(low=0, high=256, size=(n_seeds, n_keys, key_length)).astype(np.uint8)
        seeds = np.random.randint(low=0, high=2**32, size=n_seeds, dtype=np.uint32)

        # benchmark sklearn
        tic = perf_counter()
        for i in range(n_seeds):
            for j in range(n_keys):
                _ = sk_mmh3(key=keys[i, j].tobytes(), seed=seeds[i], positive=True)
        tac = perf_counter()
        elapse = tac - tic
        print(f"sklearn: {elapse: .8f}")
        sklearn_times.append(elapse)

        # benchmark my hash numpy
        tic = perf_counter()
        _ = my_mmh3(keys.view(dtype=np.uint32), seeds=seeds)
        tac = perf_counter()
        elapse = tac - tic
        print(f"numpy: {elapse: .8f}")
        my_numpy_times.append(elapse)

        # warm up gpu for once
        if not warmup:
            keys = cupy.array(keys.view(dtype=np.uint32))
            seeds = cupy.array(seeds)
            _ = my_mmh3(keys, seeds=seeds).get()
            warmup = True
        # benchmark my hash cupy
        tic = perf_counter()
        keys = cupy.array(keys.view(dtype=np.uint32))
        seeds = cupy.array(seeds)
        _ = my_mmh3(keys, seeds=seeds).get()
        tac = perf_counter()
        elapse = tac - tic
        print(f"cupy: {elapse: .8f}")
        my_cupy_times.append(elapse)

    # plot the result
    grid = np.array(list(product(n_keys_list, n_seeds_list)))
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(grid[:, 0], grid[:, 1], sklearn_times, marker="o", label="sk")
    ax.scatter(grid[:, 0], grid[:, 1], my_numpy_times, marker="+", label="numpy")
    ax.scatter(grid[:, 0], grid[:, 1], my_cupy_times, marker="^", label="cupy")
    ax.set_xlabel('#keys')
    ax.set_ylabel('#seeds')
    ax.set_zlabel('latency(s)')
    ax.legend()
    plt.savefig("result.png", dpi=200)
