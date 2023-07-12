import logging
from pathlib import Path
import jaxlib.xla_extension as jax_xla

import x_xy
from x_xy.base import System


def test_xml_parsing():
    for example_name in x_xy.io.list_examples():
        xml_path = Path(__file__).parent.joinpath(
            f"examples/{example_name}.xml")
        original_sys: x_xy.System = x_xy.io.xml.load_sys_from_xml(xml_path)

        sys_to_xml_str = x_xy.io.xml.save_sys_to_xml_str(original_sys)

        logging.debug(sys_to_xml_str)

        compare_sys = x_xy.io.xml.load_sys_from_str(sys_to_xml_str)

        assert System.deep_equal(original_sys, compare_sys)

        print(f"Passed {example_name}.xml")

    def double_load_xml_to_sys(xml_path: Path) -> System:
        orig_sys = x_xy.io.xml.load_sys_from_xml(
            Path(__file__).parent.joinpath(xml_path))
        exported_xml = x_xy.io.xml.save_sys_to_xml_str(orig_sys)
        new_sys = x_xy.io.xml.load_sys_from_str(exported_xml)
        return new_sys

    sys_test_xml_1 = double_load_xml_to_sys("examples/test_all_1.xml")
    sys_test_xml_2 = double_load_xml_to_sys("examples/test_all_2.xml")

    assert not System.deep_equal(sys_test_xml_1, sys_test_xml_2)


if __name__ == "__main__":
    test_xml_parsing()
