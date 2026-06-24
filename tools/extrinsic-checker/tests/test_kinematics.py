import os

import numpy as np
import pytest

from extrinsic_checker.kinematics import build_cfg, Kinematics

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
G1_URDF = os.path.join(REPO, "example-dataset/guodi/a2d/g1/g1_flat.urdf")


def test_build_cfg_applies_sign_and_defaults():
    mapping = {"head": {"h5_path": "x", "entries": [
        {"h5_index": 0, "urdf_joint": "jA", "sign": -1},
        {"h5_index": 1, "urdf_joint": "jB"},
    ]}}
    fv = {"head": np.array([0.5, 0.2])}
    cfg = build_cfg(fv, mapping, ["jA", "jB", "jC"])
    assert cfg == {"jA": -0.5, "jB": 0.2, "jC": 0.0}


def test_build_cfg_index_out_of_range():
    mapping = {"head": {"h5_path": "x", "entries": [{"h5_index": 5, "urdf_joint": "jA"}]}}
    with pytest.raises(ValueError):
        build_cfg({"head": np.array([0.1])}, mapping, ["jA"])


@pytest.mark.skipif(not os.path.exists(G1_URDF), reason="g1 urdf not present")
def test_fk_head_link2_matches_validated_pose():
    kin = Kinematics(G1_URDF, "base_link")
    mapping = {
        "waist": {"h5_path": "w", "entries": [
            {"h5_index": 0, "urdf_joint": "idx02_body_joint2", "sign": -1},
            {"h5_index": 1, "urdf_joint": "idx01_body_joint1", "sign": 1}]},
        "head": {"h5_path": "h", "entries": [
            {"h5_index": 0, "urdf_joint": "idx11_head_joint1", "sign": 1},
            {"h5_index": 1, "urdf_joint": "idx12_head_joint2", "sign": 1}]},
    }
    fv = {"waist": np.array([0.7083, 0.3885]), "head": np.array([0.0, 0.4363])}
    cfg = build_cfg(fv, mapping, kin.robot.actuated_joint_names)
    T = kin.link_transform(cfg, "head_link2")
    # Validated config (body_pitch NEGATED) — the one that reconstructs the
    # tabletop correctly. head_link2 sits ~1.45 m high, slightly toward -x (front).
    assert np.allclose(T[:3, 3], [-0.157, 0.0, 1.450], atol=0.01)
