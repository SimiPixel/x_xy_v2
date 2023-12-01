import jax
import jax.numpy as jnp
import numpy as np
import tree_utils

import x_xy
from x_xy.algorithms.generator.transforms import _draw_pos_uniform
from x_xy.algorithms.generator.transforms import _setup_fn_randomize_positions
from x_xy.subpkgs import sys_composer


def finalize_fn_full_imu_setup(key, q, x, sys):
    X = {
        name: x_xy.imu(x.take(sys.name_to_idx(name), 1), sys.gravity, sys.dt)
        for name in sys.link_names
    }
    return X, None


def test_normalize():
    sys = x_xy.load_example("test_three_seg_seg2")
    gen = x_xy.build_generator(sys, finalize_fn=finalize_fn_full_imu_setup)
    gen = x_xy.batch_generators_lazy(gen, 50)

    normalizer = x_xy.make_normalizer_from_generator(
        gen, approx_with_large_batchsize=50
    )
    X, _ = gen(jax.random.split(jax.random.PRNGKey(777))[1])
    X = normalizer(X)
    X_flat = tree_utils.batch_concat(X, 2)
    X_mean = jnp.mean(X_flat, (0, 1))
    X_std = jnp.std(X_flat, (0, 1))

    delta = 0.0001
    assert jnp.all(jnp.logical_and(X_mean > -delta, X_mean < delta))
    assert jnp.all(jnp.logical_and(X_std > (1 - delta), X_std < (1 + delta)))


def setup_fn_old(key, sys: x_xy.System) -> x_xy.System:
    def replace_pos(transforms, new_pos, name: str):
        i = sys.name_to_idx(name)
        return transforms.index_set(i, transforms[i].replace(pos=new_pos))

    ts = sys.links.transform1

    # seg 1 relative to seg2
    key, pos = _draw_pos_uniform(key, [-0.2, -0.02, -0.02], [-0.0, 0.02, 0.02])
    ts = replace_pos(ts, pos, "seg1")

    # imu1 relative to seg1
    key, pos = _draw_pos_uniform(key, [-0.25, -0.05, -0.05], [-0.05, 0.05, 0.05])
    ts = replace_pos(ts, pos, "imu1")

    # seg3 relative to seg2
    key, pos = _draw_pos_uniform(key, [0.0, -0.02, -0.02], [0.2, 0.02, 0.02])
    ts = replace_pos(ts, pos, "seg3")

    # seg4 relative to seg3
    key, pos = _draw_pos_uniform(key, [0.0, -0.02, -0.02], [0.4, 0.02, 0.02])
    ts = replace_pos(ts, pos, "seg4")

    # imu2 relative to seg3
    key, pos = _draw_pos_uniform(key, [0.05, -0.05, -0.05], [0.25, 0.05, 0.05])
    ts = replace_pos(ts, pos, "imu2")

    return sys.replace(links=sys.links.replace(transform1=ts))


def test_randomize_positions():
    key = jax.random.PRNGKey(1)
    sys = x_xy.load_example("test_randomize_position")

    # split key once more because the new logic `setup_fn_randomize_positions`
    # randomizes the position for each body even if the body has
    # no explicit `pos_min` and `pos_max` given in the xml
    # this is the case here for the body `seg2`
    # i.e. this is the split for `seg2` relative to `worldbody`
    internal_key, *_ = jax.random.split(key, 4)
    # then comes `seg1` relative to `seg2`
    pos_old = setup_fn_old(internal_key, sys).links.transform1.pos

    pos_new = _setup_fn_randomize_positions(key, sys).links.transform1.pos

    np.testing.assert_array_equal(pos_old, pos_new)


def test_cor():
    sys = x_xy.load_example("test_three_seg_seg2")
    sys = sys_composer.inject_system(sys, sys.add_prefix_suffix("second_"))
    x_xy.build_generator(sys, x_xy.RCMG_Config(cor=True), _compat=True)(
        jax.random.PRNGKey(1)
    )


def test_knee_flexible_imus_sim():
    sys = x_xy.load_example("knee_flexible_imus")
    qref = np.ones((101, 22))
    qref[:, :4] /= np.linalg.norm(qref[:, :4], axis=-1, keepdims=True)
    qref[:, 7:11] /= np.linalg.norm(qref[:, 7:11], axis=-1, keepdims=True)
    qref[:, 15:19] /= np.linalg.norm(qref[:, 15:19], axis=-1, keepdims=True)

    q_target = jnp.array(
        [
            5.1144928e-01,
            4.7150385e-01,
            4.8405796e-01,
            5.3084046e-01,
            8.8379657e-01,
            9.7867215e-01,
            9.7680002e-01,
            9.9995208e-01,
            -2.7399960e-03,
            9.4035007e-03,
            1.5015506e-04,
            -2.0966224e-02,
            -3.4544379e-03,
            -4.0299362e-03,
            1.0504098e00,
            9.9990761e-01,
            -3.6321122e-03,
            1.2647988e-02,
            -3.4033409e-03,
            -6.1367713e-03,
            -4.6193558e-03,
            -2.1974782e-02,
        ]
    )

    q, _ = x_xy.build_generator(
        sys,
        x_xy.RCMG_Config(T=0.1),
        imu_motion_artifacts=True,
        dynamic_simulation=True,
        dynamic_simulation_kwargs=dict(
            # back then i used `initial_sim_state_is_zeros` = False in combination
            # with `concat_configs` to create two seconds of initial nomotion phase
            # but the new logic is better (that is with initial_sim_state_... = False)
            overwrite_q_ref=qref,
            initial_sim_state_is_zeros=True,
        ),
        imu_motion_artifacts_kwargs=dict(hide_injected_bodies=False),
        _compat=True,
    )(jax.random.PRNGKey(1))

    np.testing.assert_allclose(q[-1], q_target, atol=1.2e-7, rtol=1.3e-5)
