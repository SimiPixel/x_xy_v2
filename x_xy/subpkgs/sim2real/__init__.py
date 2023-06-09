import logging
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import tree_utils

import x_xy
from x_xy import algebra, maths, scan
from x_xy.base import System, Transform


def xs_from_raw(
    sys: x_xy.base.System,
    link_name_pos_rot: dict,
    t1: float = 0.0,
    t2: Optional[float] = None,
    eps_frame: Optional[str] = None,
    qinv: bool = True,
) -> x_xy.base.Transform:
    """Build time-series of maximal coordinates `xs` from raw position and
    quaternion trajectory data. This function scans through each link (as
    defined by `sys`), looks for the raw data in `link_name_pos_rot` using
    the `link_name` as identifier. It inverts the quaternion if `qinv`.
    Then, it creates a `Transform` that transforms from epsilon (as defined
    by `eps_frame`) to the link for each timestep. Finally, it stacks all
    transforms in order as defined by `sys` along the 1-th axis. The 0-th
    axis is time axis.

    Args:
        sys (x_xy.base.System): System which defines ordering of returned `xs`
        link_name_pos_rot (dict): Dictonary of `link_name` ->
            {'pos': ..., 'quat': ...}. Obtained, e.g., using `process_omc`.
        t1 (float, optional): Crop time left. Defaults to 0.0.
        t2 (Optional[float], optional): Crop time right. Defaults to None.
        eps_frame (str, optional): Move into this segment's frame at time zero as
            eps frame. Defaults to `None`.
            If `None`: Use root-frame as eps-frame.
            If 'none': Don't move into a specific eps-frame.

    Returns:
        x_xy.base.Transform: Time-series of eps-to-link transformations
    """

    if eps_frame == "none":
        logging.warning(
            "`eps_frame` set to `none` might lead to problems with artificial IMUs,"
            " since the gravity vector is assumed to be in positive z-axis in eps-frame"
        )

    link_name_pos_rot = _crop_sequence(link_name_pos_rot, sys.dt, t1, t2)

    # determine `eps_frame` transform
    if eps_frame != "none":
        if eps_frame is None:
            connect_to_base = []
            # find link and link name that connects to world
            for link_name, link_parent in zip(sys.link_names, sys.link_parents):
                if link_parent == -1:
                    connect_to_base.append(link_name)
            assert len(connect_to_base) == 1, (
                f"Ambiguous `eps-frame` since multiple links ({connect_to_base})"
                " connect to base."
            )
            eps_frame = connect_to_base[0]
        eps = link_name_pos_rot[eps_frame]
        q_eps = eps["quat"][0]
        if qinv:
            q_eps = x_xy.maths.quat_inv(q_eps)
        t_eps = x_xy.base.Transform(eps["pos"][0], q_eps)
    else:
        t_eps = x_xy.base.Transform.zero()

    # build `xs` from optical motion capture data
    xs = []

    def f(_, __, link_name: str):
        q, pos = (
            link_name_pos_rot[link_name]["quat"],
            link_name_pos_rot[link_name]["pos"],
        )
        if qinv:
            q = x_xy.maths.quat_inv(q)
        t = x_xy.base.Transform(pos, q)
        t = x_xy.algebra.transform_mul(t, x_xy.algebra.transform_inv(t_eps))
        xs.append(t)

    x_xy.scan.tree(sys, f, "l", sys.link_names)

    # stack and permute such that time-axis is 0-th axis
    xs = xs[0].batch(*xs[1:])
    xs = xs.transpose((1, 0, 2))
    return xs


def match_xs(sys: System, xs: Transform, sys_xs: System) -> Transform:
    """Match tranforms `xs` to subsystem `sys`.

    Args:
        sys (System): Smaller system. Every link in `sys` must be in `sys_xs`.
        xs (Transform): Transforms of larger system.
        sys_xs (Transform): Larger system.

    Returns:
        Transform: Transforms of smaller system.
    """
    _checks_time_series_of_xs(sys_xs, xs)

    # disable logging temporarily because otherwise it will warn because of the usage
    # of `eps_frame` = 'none'
    logging.disable(logging.WARNING)
    xs_small = xs_from_raw(
        sys,
        {
            name: {
                "pos": xs.pos[:, sys_xs.name_to_idx(name)],
                "quat": xs.rot[:, sys_xs.name_to_idx(name)],
            }
            for name in sys_xs.link_names
        },
        eps_frame="none",
        qinv=False,
    )
    logging.disable(logging.INFO)
    return xs_small


def unzip_xs(sys: System, xs: Transform) -> Tuple[Transform, Transform]:
    """Split eps-to-link transforms into parent-to-child pure
    translational `transform1` and pure rotational `transform2`.

    Args:
        sys (System): Defines scan.tree
        xs (Transform): Eps-to-link transforms

    Returns:
        Tuple[Transform, Transform]: transform1, transform2
    """
    _checks_time_series_of_xs(sys, xs)

    @jax.vmap
    def _unzip_xs(xs):
        def f(_, __, i: int, p: int):
            if p == -1:
                x_parent_to_link = xs[i]
            else:
                x_parent_to_link = algebra.transform_mul(
                    xs[i], algebra.transform_inv(xs[p])
                )

            transform1_pos = Transform.create(pos=x_parent_to_link.pos)
            transform2_rot = Transform.create(rot=x_parent_to_link.rot)
            return (transform1_pos, transform2_rot)

        return scan.tree(sys, f, "ll", list(range(sys.num_links())), sys.link_parents)

    return _unzip_xs(xs)


def zip_xs(
    sys: System,
    xs_transform1: Transform,
    xs_transform2: Transform,
) -> Transform:
    """Performs forward kinematics using `transform1` and `transform2`.

    Args:
        sys (x_xy.base.System): Defines scan.tree
        xs_transform1 (x_xy.base.Transform): Applied before `transform1`
        xs_transform2 (x_xy.base.Transform): Applied after `transform2`

    Returns:
        x_xy.base.Transform: Time-series of eps-to-link transformations
    """
    _checks_time_series_of_xs(sys, xs_transform1)
    _checks_time_series_of_xs(sys, xs_transform2)

    @jax.vmap
    def _zip_xs(xs_transform1, xs_transform2):
        eps_to_l = {-1: x_xy.base.Transform.zero()}

        def f(_, __, i: int, p: int):
            transform = algebra.transform_mul(xs_transform2[i], xs_transform1[i])
            eps_to_l[i] = algebra.transform_mul(transform, eps_to_l[p])
            return eps_to_l[i]

        return x_xy.scan.tree(
            sys, f, "ll", list(range(sys.num_links())), sys.link_parents
        )

    return _zip_xs(xs_transform1, xs_transform2)


def _checks_time_series_of_xs(sys, xs):
    assert tree_utils.tree_ndim(xs) == 3, f"pos.shape={xs.pos.shape}"
    num_links_xs, num_links_sys = tree_utils.tree_shape(xs, axis=1), sys.num_links()
    assert num_links_xs == num_links_sys, f"{num_links_xs} != {num_links_sys}"


def delete_to_world_pos_rot(sys: System, xs: Transform) -> Transform:
    """Replace the transforms of all links that connect to the worldbody
    by unity transforms.

    Args:
        sys (System): System only used for structure (in scan.tree).
        xs (Transform): Time-series of transforms to be modified.

    Returns:
        Transform: Time-series of modified transforms.
    """
    _checks_time_series_of_xs(sys, xs)

    @jax.vmap
    def _delete_to_world_pos_rot(xs):
        def f(_, __, i: int, p: int):
            x_link = xs[i]
            if p == -1:
                x_link = Transform.zero()
            return x_link

        return scan.tree(
            sys,
            f,
            "ll",
            list(range(sys.num_links())),
            sys.link_parents,
        )

    return _delete_to_world_pos_rot(xs)


def scale_xs(
    sys: System,
    xs: Transform,
    factor: float,
    exclude: list[str] = ["px", "py", "pz", "free"],
) -> Transform:
    """Increase / decrease transforms by scaling their positional / rotational
    components based on the systems link type, i.e. the `xs` should conceptionally
    be `transform2` objects.

    Args:
        sys (System): System defining structure (for scan.tree)
        xs (Transform): Time-series of transforms to be modified.
        factor (float): Multiplicative factor.
        exclude (list[str], optional): Skip scaling of transforms if their link_type
            is one of those. Defaults to ["px", "py", "pz", "free"].

    Returns:
        Transform: Time-series of scaled transforms.
    """
    _checks_time_series_of_xs(sys, xs)

    @jax.vmap
    def _scale_xs(xs):
        def f(_, __, i: int, type: str):
            x_link = xs[i]
            if type not in exclude:
                x_link = _scale_transform_based_on_type(x_link, type, factor)
            return x_link

        return scan.tree(sys, f, "ll", list(range(sys.num_links())), sys.link_types)

    return _scale_xs(xs)


def _scale_transform_based_on_type(x: Transform, link_type: str, factor: float):
    pos, rot = x.pos, x.rot
    if link_type in ["px", "py", "pz", "free"]:
        pos = pos * factor
    if link_type in ["rx", "ry", "rz", "spherical", "free"]:
        axis, angle = maths.quat_to_rot_axis(rot)
        rot = maths.quat_rot_axis(axis, angle * factor)
    return Transform(pos, rot)


def _crop_sequence(data: dict, dt: float, t1: float = 0.0, t2: Optional[float] = None):
    # crop time left and right
    if t2 is None:
        t2i = tree_utils.tree_shape(data)
    else:
        t2i = int(t2 / dt)
    t1i = int(t1 / dt)
    return jax.tree_map(lambda arr: jnp.array(arr)[t1i:t2i], data)
