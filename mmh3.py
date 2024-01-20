""" murmurhash3 implementation compatible with numpy/cupy """
from typing import Protocol, Tuple, Union, runtime_checkable


@runtime_checkable
class UINT32NDArray(Protocol):
    """  cupy/numpy uint32 ndarray or other compatible """
    shape: Tuple[int, ...]
    ndim: int
    size: int

    def copy(self) -> "UINT32NDArray":
        """ deep copy """

    def __getitem__(self, *args, **kwargs) -> "UINT32NDArray":
        ...

    def __iadd__(self, x: Union["UINT32NDArray", int]) -> "UINT32NDArray":
        ...

    def __imul__(self, x: int) -> "UINT32NDArray":
        ...

    def __ixor__(self, x: Union["UINT32NDArray", int]) -> "UINT32NDArray":
        ...

    def __ilshift__(self, x: int) -> "UINT32NDArray":
        ...

    def __irshift__(self, x: int) -> "UINT32NDArray":
        ...

    def __lshift__(self, x: int) -> "UINT32NDArray":
        ...

    def __rshift__(self, x: int) -> "UINT32NDArray":
        ...

    def __or__(self, x: "UINT32NDArray") -> "UINT32NDArray":
        ...

    def __xor__(self, x: "UINT32NDArray") -> "UINT32NDArray":
        ...



def _rotl(x: UINT32NDArray, r: int) -> UINT32NDArray:
    assert 0 <= r <= 32
    return (x << r) | (x >> (32 - r))


def _fmix(h: UINT32NDArray) -> UINT32NDArray:
    h ^= h >> 16
    h *= 2246822507  # int("85ebca6b", 16)
    h ^= h >> 13
    h *= 3266489909  # int("c2b2ae35", 16)
    h ^= h >> 16
    return h


def murmurhash3_32(keys: UINT32NDArray, seeds: UINT32NDArray) -> UINT32NDArray:
    """murmurhash3 with 32-bit data

    The input/output can be numpy/cupy uint32 ndarrays or other objects with compatible interface.
    If you need to hash other objects other than uint32:

    (1) convert it to bytes, e.g. `k="somehash".encode()`
    (2) make sure the bytes length is a multiplier of 4: `assert len(k) % 4 == 0`
    (3) conver it to uint32 ndarray, e.g. `k=cupy.frombuffer(k, dtype=cupy.uint32)`

    Args:
        keys (UINT32NDArray): 3-dim uint32 array to be hashed, shape: (n_seeds, n_keys, key_length)
        seeds (UINT32NDArray): 1-dim uint32 array seeds, shape: (n_seeds, ),
            each seed is mapped to each sub-array along the 0-th dim of keys.

    Returns:
        UINT32NDArray: 1-dim uint32 hash values, shape: (n_seeds, n_keys)
    """
    assert keys.ndim == 3
    n_seeds, n_keys, key_length = keys.shape
    assert 1 <= key_length <= 4294967295  # int("00000000FFFFFFFF", 16)
    assert seeds.ndim == 1 and seeds.size == n_seeds

    # body (all operations are element-wise)
    keys = keys.copy()
    c1 = 3432918353  # int("cc9e2d51", 16)
    c2 = 461845907  # int("1b873593", 16)
    keys *= c1
    keys = _rotl(keys, 15)
    keys *= c2

    h1 = keys[..., 0] ^ seeds[:, None]  # shape = (n_seeds, n_keys)
    h1 = _rotl(h1, 13)
    h1 *= 5
    h1 += 3864292196  # int("e6546b64", 16)
    for i in range(1, key_length):
        h1 ^= keys[..., i]
        h1 = _rotl(h1, 13)
        h1 *= 5
        h1 += 3864292196  # int("e6546b64", 16)

    # finalization
    h1 ^= (4 * key_length)
    h1 = _fmix(h1)
    assert h1.shape == (n_seeds, n_keys)
    return h1
