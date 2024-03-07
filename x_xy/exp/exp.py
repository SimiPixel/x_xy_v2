from functools import cache
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import joblib
import ring
from ring.exp import omc_utils
import tree_utils
import yaml

arm_xml = "setups/arm.xml"
gait_xml = "setups/gait.xml"
_id2xml = {
    "S_04": arm_xml,
    "S_06": arm_xml,
    "S_07": arm_xml,
    "S_08": arm_xml,
    "S_09": arm_xml,
    "S_10": arm_xml,
    "S_12": gait_xml,
    "S_13": gait_xml,
    "S_14": gait_xml,
    "S_15": gait_xml,
    "S_16": gait_xml,
    "T_01": gait_xml,
}


def _relative_to_this_file(path: str) -> Path:
    return Path(__file__).parent.joinpath(path)


def _read_yaml(path: str):
    with open(_relative_to_this_file(path)) as file:
        yaml_str = yaml.safe_load(file)
    return yaml_str


@cache
def load_sys(
    exp_id: str,
) -> ring.base.System:
    xml_path = _relative_to_this_file(_id2xml[exp_id])
    sys = ring.io.load_sys_from_xml(xml_path)
    return sys


@cache
def load_data(
    exp_id: str,
    motion_start: Optional[str] = None,
    motion_stop: Optional[str] = None,
    left_padd: float = 0.0,
    right_padd: float = 0.0,
    resample_to_hz: float = 100.0,
) -> dict:
    trial_data = joblib.load(ring.utils.download_from_repo(f"data/{exp_id}.joblib"))

    metadata = _read_yaml("metadata.yaml")[exp_id]
    timings = metadata["timings"]
    hz_imu, hz_omc = float(metadata["hz"]["imu"]), float(metadata["hz"]["omc"])

    trial_data = omc_utils.resample(
        trial_data,
        hz_in=omc_utils.hz_helper(trial_data.keys(), hz_imu=hz_imu, hz_omc=hz_omc),
        hz_out=resample_to_hz,
        vecinterp_method="cubic",
    )
    trial_data = omc_utils.crop_tail(
        trial_data, resample_to_hz, strict=True, verbose=False
    )

    if motion_start is not None:
        assert (
            motion_start in timings
        ), f"`{motion_start}` is not one of {load_timings(exp_id).keys()}"

        motion_sequence = list(timings.keys())
        next_motion_i = motion_sequence.index(motion_start) + 1
        assert next_motion_i < len(motion_sequence)

        if motion_stop is None:
            motion_stop = motion_sequence[next_motion_i]

        assert (
            motion_stop in timings
        ), f"`{motion_stop}` is not one of {load_timings(exp_id).keys()}"

        assert motion_sequence.index(motion_start) < motion_sequence.index(
            motion_stop
        ), "Empty sequence, stop <= start"

        t1 = timings[motion_start] - left_padd
        # ensure that t1 >= 0
        t1 = max(t1, 0.0)
        t2 = timings[motion_stop] + right_padd

        trial_data = _crop_sequence(trial_data, 1 / resample_to_hz, t1=t1, t2=t2)
    else:
        assert motion_stop is None

    return trial_data


def load_timings(exp_id: str) -> dict[str, float]:
    return _read_yaml("metadata.yaml")[exp_id]["timings"]


def _crop_sequence(data: dict, dt: float, t1: float = 0.0, t2: Optional[float] = None):
    # crop time left and right
    if t2 is None:
        t2i = tree_utils.tree_shape(data)
    else:
        t2i = int(t2 / dt)
    t1i = int(t1 / dt)
    return jax.tree_map(lambda arr: jnp.array(arr)[t1i:t2i], data)
