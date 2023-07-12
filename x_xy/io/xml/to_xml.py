import jax.numpy as jnp


from x_xy import base
from x_xy.io.xml import abstract
from x_xy.io.xml.abstract import _to_str
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom.minidom import parseString


def save_sys_to_xml_str(sys: base.System) -> str:
    global_index_map = {qd: sys.idx_map(qd) for qd in ["q", "d"]}

    # Create root element
    x_xy = Element("x_xy")
    x_xy.set("model", sys.model_name)

    options = SubElement(x_xy, "options")
    options.set("dt", str(sys.dt))
    options.set("gravity", _to_str(sys.gravity))

    # Create worldbody
    worldbody = SubElement(x_xy, "worldbody")

    def process_link(link_idx: int, parent_elem: Element):
        link = sys.links[link_idx]
        link_typ = sys.link_types[link_idx]
        link_name = sys.link_names[link_idx]

        # Create body element
        body = SubElement(parent_elem, "body")
        body.set("joint", link_typ)
        body.set("name", link_name)

        # Set attributes
        abstract.AbsTrans.to_xml(body, link.transform1)
        abstract.AbsPosMinMax.to_xml(body, link.pos_min, link.pos_max)
        abstract.AbsDampArmaStiffZero.to_xml(
            body,
            sys.link_damping[global_index_map["d"][link_name]],
            sys.link_armature[global_index_map["d"][link_name]],
            sys.link_spring_stiffness[global_index_map["d"][link_name]],
            sys.link_spring_zeropoint[global_index_map["d"][link_name]],
            base.Q_WIDTHS[link_typ],
            base.QD_WIDTHS[link_typ],
            link_typ,
        )

        # Add geometry elements
        geoms = sys.geoms
        for geom in geoms:
            if geom.link_idx == link_idx:
                geom_elem = SubElement(body, "geom")
                abstract_class = abstract.geometry_to_abstract[type(geom)]
                abstract_class.to_xml(geom_elem, geom)

        # Recursively process child links
        for child_idx, parent_idx in enumerate(sys.link_parents):
            if parent_idx == link_idx:
                process_link(child_idx, body)

    for root_link_idx, parent_idx in enumerate(sys.link_parents):
        if parent_idx == -1:
            process_link(root_link_idx, worldbody)

    # Pretty print xml
    xml_str = parseString(tostring(x_xy)).toprettyxml(indent="  ")
    return xml_str


def save_sys_to_xml(sys: base.System, xml_path: str) -> None:
    xml_str = save_sys_to_xml_str(sys)
    with open(xml_path, "w") as f:
        f.write(xml_str)
