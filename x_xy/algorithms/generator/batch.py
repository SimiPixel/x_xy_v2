import random
import warnings

import jax
import jax.numpy as jnp
from tqdm import tqdm
import tree_utils
from tree_utils import tree_batch

from x_xy import utils

from .types import BatchedGenerator
from .types import Generator


def _build_batch_matrix(batchsizes: list[int]) -> jax.Array:
    arr = []
    for i, l in enumerate(batchsizes):
        arr += [i] * l
    return jnp.array(arr)


def batch_generators_lazy(
    generators: Generator | list[Generator],
    batchsizes: int | list[int] = 1,
    stochastic: bool = False,
) -> BatchedGenerator:
    """Create a large generator by stacking multiple generators lazily.
    NOTE: If `stochastic` then `batchsizes` must be a single integer.
    """

    if not isinstance(generators, list):
        # test if generator is already batched, then this is a no-op
        key = jax.random.PRNGKey(0)
        X, *_ = generators(key)
        ndim = tree_utils.tree_ndim(X)
        if ndim > 2:
            warnings.warn(f"`generators` seem already batched. ndim={ndim}")
            return generators

    generators = utils.to_list(generators)

    if stochastic:
        assert isinstance(batchsizes, int)
        bs_total = batchsizes
        pmap, vmap = utils.distribute_batchsize(bs_total)
    else:
        batchsizes = utils.to_list(batchsizes)
        assert len(generators) == len(batchsizes)

        batch_arr_nonstoch = _build_batch_matrix(batchsizes)
        bs_total = len(batch_arr_nonstoch)
        pmap, vmap = utils.distribute_batchsize(bs_total)
        batch_arr_nonstoch = batch_arr_nonstoch.reshape((pmap, vmap))

    pmap_trafo = jax.pmap
    # single GPU node, then do jit + vmap instead of pmap
    # this allows e.g. better NAN debugging capabilities
    if pmap == 1:
        pmap_trafo = lambda f: jax.jit(jax.vmap(f))

    @pmap_trafo
    @jax.vmap
    def _generator(key, which_gen: int):
        return jax.lax.switch(which_gen, generators, key)

    def generator(key):
        if stochastic:
            key, consume = jax.random.split(key)
            batch_arr = jax.random.choice(
                consume, jnp.arange(len(generators)), shape=(pmap, vmap)
            )
        else:
            batch_arr = batch_arr_nonstoch

        pmap_vmap_keys = jax.random.split(key, bs_total).reshape((pmap, vmap, 2))
        data = _generator(pmap_vmap_keys, batch_arr)

        # merge pmap and vmap axis
        data = utils.merge_batchsize(data, pmap, vmap)

        return data

    return generator


def batch_generators_eager_to_list(
    generators: Generator | list[Generator],
    sizes: int | list[int],
    seed: int = 1,
    transfer_to_cpu: bool = True,
) -> list[tree_utils.PyTree]:
    "Returns list of unbatched sequences."
    generators, sizes = utils.to_list(generators), utils.to_list(sizes)
    assert len(generators) == len(sizes)

    key = jax.random.PRNGKey(seed)
    data = []
    for gen, size in tqdm(zip(generators, sizes), desc="eager data generation"):
        key, consume = jax.random.split(key)
        sample = batch_generators_lazy(gen, size)(consume)
        if transfer_to_cpu:
            sample = jax.device_put(sample, jax.devices("cpu")[0])
        data.extend([jax.tree_map(lambda a: a[i], sample) for i in range(size)])
    return data


def batched_generator_from_list(
    data: list,
    batchsize: int,
    shuffle: bool = True,
    drop_last: bool = True,
    seed: int = 1,
) -> BatchedGenerator:
    assert drop_last, "Not `drop_last` is currently not implemented."
    assert len(data) >= batchsize

    N, i = len(data) // batchsize, 0
    random.seed(seed)

    def generator(key: jax.Array):
        nonlocal i
        del key
        if shuffle and i == 0:
            random.shuffle(data)

        start, stop = i * batchsize, (i + 1) * batchsize
        batch = tree_batch(data[start:stop], backend="jax")
        i = (i + 1) % N
        return batch

    return generator


def batch_generators_eager(
    generators: Generator | list[Generator],
    sizes: int | list[int],
    batchsize: int,
    shuffle: bool = True,
    drop_last: bool = True,
    seed: int = 1,
    transfer_to_cpu: bool = True,
) -> BatchedGenerator:
    """Eagerly create a large precomputed generator by calling multiple generators
    and stacking their output."""

    data = batch_generators_eager_to_list(
        generators, sizes, seed=seed, transfer_to_cpu=transfer_to_cpu
    )
    return batched_generator_from_list(data, batchsize, shuffle, drop_last, seed=seed)