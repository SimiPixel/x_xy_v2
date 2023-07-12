from xml.etree import ElementTree

import jax.numpy as jnp

import x_xy
from x_xy import base
from x_xy.io.xml import abstract


def _find_assert_unique(tree: ElementTree, *keys):
    assert len(keys) > 0

    value = tree.findall(keys[0])
    if len(value) == 0:
        return None

    assert len(value) == 1

    if len(keys) == 1:
        return value[0]
    else:
        return _find_assert_unique(value[0], *keys[1:])


def _build_defaults_attributes(tree):
    tags = ["geom", "body"]
    default_attrs = {}
    for tag in tags:
        defaults_subtree = _find_assert_unique(tree, "defaults", tag)
        if defaults_subtree is None:
            attrs = {}
        else:
            attrs = defaults_subtree.attrib
        default_attrs[tag] = attrs
    return default_attrs


def _assert_all_tags_attrs_valid(xml_tree):
    valid_attrs = {
        "x_xy": ["model"],
        "options": ["gravity", "dt"],
        "defaults": ["geom", "body"],
        "worldbody": [],
        "body": [
            "name",
            "pos",
            "pos_min",
            "pos_max",
            "quat",
            "euler",
            "joint",
            "armature",
            "damping",
            "spring_stiff",
            "spring_zero",
        ],
        "geom": ["type", "mass", "pos", "dim", "quat", "euler"],
    }
    for subtree in xml_tree.iter():
        assert subtree.tag in list([key for key in valid_attrs])
        for attr in subtree.attrib:
            if subtree.tag == "geom" and attr.split("_")[0] == "vispy":
                continue
            assert attr in valid_attrs[subtree.tag]


def _mix_in_defaults(worldbody, default_attrs):
    for subtree in worldbody.iter():
        if subtree.tag not in ["body", "geom"]:
            continue
        tag = subtree.tag
        attr = subtree.attrib
        for default_attr in default_attrs[tag]:
            if default_attr not in attr:
                attr.update({default_attr: default_attrs[tag][default_attr]})


def _convert_attrs_to_arrays(xml_tree):
    for subtree in xml_tree.iter():
        for k, v in subtree.attrib.items():
            try:
                array = [float(num) for num in v.split(" ")]
            except:  # noqa: E722
                continue
            subtree.attrib[k] = jnp.squeeze(jnp.array(array))


def _extract_geoms_from_body_xml(body, current_link_idx):
    link_geoms = []
    for geom_subtree in body.findall("geom"):
        attr = geom_subtree.attrib
        geom = abstract.xml_identifier_to_abstract[attr["type"]].from_xml(
            attr, current_link_idx
        )
        link_geoms.append(geom)
    return link_geoms


def _initial_setup(xml_tree):
    _assert_all_tags_attrs_valid(xml_tree)
    _convert_attrs_to_arrays(xml_tree)
    default_attrs = _build_defaults_attributes(xml_tree)
    worldbody = _find_assert_unique(xml_tree, "worldbody")
    _mix_in_defaults(worldbody, default_attrs)
    return worldbody


DEFAULT_GRAVITY = jnp.array([0, 0, 9.81])
DEFAULT_DT = jnp.array(0.01, dtype=float)


def load_sys_from_str(xml_str: str, prefix: str = ""):
    xml_tree = ElementTree.fromstring(xml_str)
    worldbody = _initial_setup(xml_tree)

    # check that <x_xy model="..."> syntax is correct
    assert xml_tree.tag == "x_xy", (
        "The root element in the xml of a x_xy model must be `x_xy`."
        " Look up the examples under  x_xy/io/examples/*.xml to get started"
    )
    model_name = xml_tree.attrib.get("model", None)
    options = _find_assert_unique(xml_tree, "options")
    options = (
        {"gravity": DEFAULT_GRAVITY, "dt": DEFAULT_DT}
        if options is None
        else options.attrib
    )

    links = {}
    link_parents = {}
    link_names = {}
    link_types = {}
    geoms = {}
    armatures = {}
    dampings = {}
    spring_stiffnesses = {}
    spring_zeropoints = {}
    global_link_idx = -1

    def process_body(body: ElementTree, parent: int):
        nonlocal global_link_idx
        global_link_idx += 1
        current_link_idx = global_link_idx
        current_link_typ = body.attrib["joint"]

        link_parents[current_link_idx] = parent
        link_types[current_link_idx] = current_link_typ
        link_names[current_link_idx] = prefix + body.attrib["name"]

        transform = abstract.AbsTrans.from_xml(body.attrib)
        pos_min, pos_max = abstract.AbsPosMinMax.from_xml(body.attrib, transform.pos)
        links[current_link_idx] = base.Link(transform, pos_min, pos_max)

        q_size = base.Q_WIDTHS[current_link_typ]
        qd_size = base.QD_WIDTHS[current_link_typ]

        (
            damping,
            armature,
            stiffness,
            zeropoint,
        ) = abstract.AbsDampArmaStiffZero.from_xml(
            body.attrib, q_size, qd_size, current_link_typ
        )

        armatures[current_link_idx] = armature
        dampings[current_link_idx] = damping
        spring_stiffnesses[current_link_idx] = stiffness
        spring_zeropoints[current_link_idx] = zeropoint

        geoms[current_link_idx] = _extract_geoms_from_body_xml(body, current_link_idx)

        for subbodies in body.findall("body"):
            process_body(subbodies, current_link_idx)

        return

    for body in worldbody.findall("body"):
        process_body(body, -1)

    def assert_order_then_to_list(d: dict) -> list:
        assert [i for i in d] == list(range(len(d)))
        return [d[i] for i in d]

    links = assert_order_then_to_list(links)
    links = links[0].batch(*links[1:])
    dampings = jnp.concatenate(assert_order_then_to_list(dampings))
    armatures = jnp.concatenate(assert_order_then_to_list(armatures))
    spring_stiffnesses = jnp.concatenate(assert_order_then_to_list(spring_stiffnesses))
    spring_zeropoints = jnp.concatenate(assert_order_then_to_list(spring_zeropoints))

    # add all geoms directly connected to worldbody
    flat_geoms = [geom for geoms in assert_order_then_to_list(geoms) for geom in geoms]
    flat_geoms += _extract_geoms_from_body_xml(worldbody, -1)

    sys = base.System(
        assert_order_then_to_list(link_parents),
        links,
        assert_order_then_to_list(link_types),
        dampings,
        armatures,
        spring_stiffnesses,
        spring_zeropoints,
        options["dt"],
        False,
        flat_geoms,
        options["gravity"],
        link_names=assert_order_then_to_list(link_names),
        model_name=model_name,
    )

    return x_xy.io.parse_system(sys)


def load_sys_from_xml(xml_path: str, prefix: str = ""):
    return load_sys_from_str(_load_xml(xml_path), prefix=prefix)


def _load_xml(xml_path: str) -> str:
    with open(xml_path, "r") as f:
        xml_str = f.read()
    return xml_str
