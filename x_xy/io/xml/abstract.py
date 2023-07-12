from typing import Tuple, TypeVar

import jax
import jax.numpy as jnp
import numpy as np

from x_xy import base

T = TypeVar("T")
ATTR = dict

default_quat = jnp.array([1.0, 0, 0, 0])
default_pos = jnp.zeros((3,))
default_damping = lambda qd_size, **_: jnp.zeros((qd_size,))
default_armature = lambda qd_size, **_: jnp.zeros((qd_size,))
default_stiffness = lambda qd_size, **_: jnp.zeros((qd_size,))


def default_zeropoint(qd_size, link_typ: str, **_):
    zeropoint = jnp.zeros((qd_size))
    if link_typ == "spherical" or link_typ == "free":
        # zeropoint then is unit quaternion and not zeros
        zeropoint = zeropoint.at[0].set(1.0)
    return zeropoint


default_fns = {
    "damping": default_damping,
    "armature": default_armature,
    "spring_stiff": default_stiffness,
    "spring_zero": default_zeropoint,
}


class AbsDampArmaStiffZero:
    @staticmethod
    def from_xml(attr: ATTR, q_size: int, qd_size: int, link_typ: str) -> list:
        return [
            jnp.atleast_1d(
                attr.get(
                    key,
                    default_fns[key](q_size=q_size, qd_size=qd_size, link_typ=link_typ),
                )
            )
            for key in ["damping", "armature", "spring_stiff", "spring_zero"]
        ]

    @staticmethod
    def to_xml(
        element: T,
        damping: jax.Array,
        armature: jax.Array,
        stiffness: jax.Array,
        zeropoint: jax.Array,
        q_size: int,
        qd_size: int,
        link_typ: str,
    ):
        for key, arr in zip(
            ["damping", "armature", "spring_stiff", "spring_zero"],
            [damping, armature, stiffness, zeropoint],
        ):
            if not _arr_equal(arr, default_fns[key](q_size=q_size, qd_size=qd_size, link_typ=link_typ)):
                element.set(key, _to_str(arr))


class AbsTrans:
    @staticmethod
    def from_xml(attr: ATTR) -> base.Transform:
        pos = attr.get("pos", default_pos)
        rot = _get_rotation(attr)
        return base.Transform(pos, rot)

    @staticmethod
    def to_xml(element: T, t: base.Transform) -> None:
        if not _arr_equal(t.pos, default_pos):
            element.set("pos", _to_str(t.pos))
        if not _arr_equal(t.rot, default_quat):
            element.set("quat", _to_str(t.rot))


class AbsPosMinMax:
    @staticmethod
    def from_xml(attr: ATTR, pos: jax.Array) -> Tuple[jax.Array, jax.Array]:
        pos_min = attr.get("pos_min", None)
        pos_max = attr.get("pos_max", None)
        assert (pos_min is None and pos_max is None) or (
            pos_min is not None and pos_max is not None
        ), (
            f"In link {attr.get('name', 'None')} found only one of `pos_min` "
            "and `pos_max`, but requires either both or none"
        )
        if pos_min is not None:
            assert not _arr_equal(
                pos_min, pos_max
            ), f"In link {attr.get('name', 'None')} "
            " both `pos_min` and `pos_max` are identical, use `pos` instead."

        if pos_min is None:
            pos_min = pos_max = pos
        return pos_min, pos_max

    @staticmethod
    def to_xml(element: T, pos_min: jax.Array, pos_max: jax.Array):
        if _arr_equal(pos_min, pos_max):
            return

        element.set("pos_min", _to_str(pos_min))
        element.set("pos_max", _to_str(pos_max))


def _from_xml_vispy(attr: ATTR):
    "Find all keys starting with `vispy_`, and return subdict without that prefix"

    def delete_prefix(key):
        len_suffix = len(key.split("_")[0]) + 1
        return key[len_suffix:]

    return {delete_prefix(k): attr[k] for k in attr if k.split("_")[0] == "vispy"}


def _to_xml_vispy(element: T, geom: base.Geometry) -> None:
    "Copy pasted from you"
    # Add vispy kwargs if they exist
    if hasattr(geom, ("vispy_kwargs")):
        for key, value in geom.vispy_kwargs.items():
            element.set(f"vispy_{key}", _to_str(value))


def _from_xml_geom_attr_processing(geom_attr: ATTR):
    "Common processing used by all geometries"
    m = geom_attr["mass"]
    t = AbsTrans.from_xml(geom_attr)
    vispy = _from_xml_vispy(geom_attr)
    return m, t, vispy


def _to_xml_geom_processing(element: T, geom: base.Geometry) -> None:
    "Common processing used by all geometries"
    AbsTrans.to_xml(element, geom.transform)
    element.set("mass", _to_str(geom.mass))
    _to_xml_vispy(element, geom)
    element.set("type", geometry_to_xml_identifier[type(geom)])


class AbsGeomBox:
    xml_geom_type: str = "box"
    geometry: base.Geometry = base.Box

    @staticmethod
    def from_xml(geom_attr: ATTR, link_idx: int) -> base.Box:
        m, t, vispy = _from_xml_geom_attr_processing(geom_attr)
        dims = [geom_attr["dim"][i] for i in range(3)]
        return base.Box(m, t, link_idx, *dims, vispy)

    @staticmethod
    def to_xml(element: T, geom: base.Box) -> None:
        _to_xml_geom_processing(element, geom)
        dim = np.array([geom.dim_x, geom.dim_y, geom.dim_z])
        element.set("dim", _to_str(dim))


class AbsGeomSphere:
    xml_geom_type: str = "sphere"
    geometry: base.Geometry = base.Sphere

    @staticmethod
    def from_xml(geom_attr: ATTR, link_idx: int) -> base.Sphere:
        m, t, vispy = _from_xml_geom_attr_processing(geom_attr)
        radius = geom_attr["dim"][0]
        return base.Sphere(m, t, link_idx, radius, vispy)

    @staticmethod
    def to_xml(element: T, geom: base.Sphere) -> None:
        _to_xml_geom_processing(element, geom)
        dim = np.array([geom.radius])
        element.set("dim", _to_str(dim))


class AbsGeomCylinder:
    xml_geom_type: str = "cylinder"
    geometry: base.Geometry = base.Cylinder

    @staticmethod
    def from_xml(geom_attr: ATTR, link_idx: int) -> base.Cylinder:
        m, t, vispy = _from_xml_geom_attr_processing(geom_attr)
        dims = [geom_attr["dim"][i] for i in range(2)]
        return base.Cylinder(m, t, link_idx, *dims, vispy)

    @staticmethod
    def to_xml(element: T, geom: base.Cylinder) -> None:
        _to_xml_geom_processing(element, geom)
        dim = np.array([geom.radius, geom.length])
        element.set("dim", _to_str(dim))


class AbsGeomCapsule:
    xml_geom_type: str = "capsule"
    geometry: base.Geometry = base.Capsule

    @staticmethod
    def from_xml(geom_attr: ATTR, link_idx: int) -> base.Capsule:
        m, t, vispy = _from_xml_geom_attr_processing(geom_attr)
        dims = [geom_attr["dim"][i] for i in range(2)]
        return base.Capsule(m, t, link_idx, *dims, vispy)

    @staticmethod
    def to_xml(element: T, geom: base.Capsule) -> None:
        _to_xml_geom_processing(element, geom)
        dim = np.array([geom.radius, geom.length])
        element.set("dim", _to_str(dim))


_ags = [AbsGeomBox, AbsGeomSphere, AbsGeomCylinder, AbsGeomCapsule]
geometry_to_xml_identifier = {ag.geometry: ag.xml_geom_type for ag in _ags}
xml_identifier_to_abstract = {ag.xml_geom_type: ag for ag in _ags}
geometry_to_abstract = {ag.geometry: ag for ag in _ags}


def _arr_equal(a, b):
    return np.all(np.array_equal(a, b))


def _get_rotation(attr: ATTR):
    rot = attr.get("quat", None)
    if rot is not None:
        assert "euler" not in attr, "Can't specify both `quat` and `euler` in xml"
    elif "euler" in attr:
        # we use zyx convention but angles are given
        # in x, y, z in the xml file
        # thus flip the order
        euler_xyz = jnp.deg2rad(attr["euler"])
        rot = base.maths.quat_euler(jnp.flip(euler_xyz), convention="zyx")
    else:
        rot = default_quat
    return rot


def _to_str(obj):
    if isinstance(obj, (np.ndarray, jnp.ndarray)):
        if obj.ndim == 0:
            return str(obj)
        return " ".join([str(x) for x in obj])
    else:
        return str(obj)
