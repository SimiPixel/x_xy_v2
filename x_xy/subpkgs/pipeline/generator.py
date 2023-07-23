import jax

import x_xy
from x_xy.algorithms import RCMG_Config
from x_xy.base import System
from x_xy.subpkgs import pipeline


def _to_list(obj):
    if not isinstance(obj, list):
        return [obj]
    return obj


def make_generator(
    configs: RCMG_Config | list[RCMG_Config],
    bs: int,
    sys_data: System | list[System],
    sys_noimu: System,
    imu_attachment: dict,
):
    configs, sys_data = _to_list(configs), _to_list(sys_data)

    def _make_generator(sys, config):
        def finalize_fn(key, q, x, sys):
            X = pipeline.imu_data(key, x, sys, imu_attachment)
            y = x_xy.algorithms.rel_pose(sys_noimu, x, sys)
            return X, y

        def setup_fn(key, sys):
            key, consume = jax.random.split(key)
            sys = x_xy.algorithms.setup_fn_randomize_positions(consume, sys)
            key, consume = jax.random.split(key)
            sys = x_xy.algorithms.setup_fn_randomize_joint_axes(consume, sys)
            return sys

        return x_xy.algorithms.build_generator(
            sys,
            config,
            setup_fn,
            finalize_fn,
        )

    gens = []
    for sys in sys_data:
        for config in configs:
            gens.append(_make_generator(sys, config))

    batchsizes = len(gens) * [bs // len(gens)]
    return x_xy.algorithms.batch_generator(gens, batchsizes)