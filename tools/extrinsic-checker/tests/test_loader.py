import json
import pytest

from extrinsic_checker.loader import load_config


def _write(tmp_path, cfg, urdf_name="r.urdf"):
    (tmp_path / urdf_name).write_text("<robot name='r'></robot>")
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg))
    return str(p)


def _good(urdf_name="r.urdf"):
    return {
        "urdf": urdf_name,
        "base_link": "base_link",
        "base_forward_axis": "-x",
        "joint_mapping": {},
        "cameras": {"head": {"mount_link": "head_link2", "modality": "depth"}},
        "thresholds": {"plane_vertical_min": 0.85, "table_height_range": [0.3, 1.2]},
    }


def test_load_valid(tmp_path):
    cfg = load_config(_write(tmp_path, _good()))
    assert cfg["base_forward_axis"] == "-x"
    assert cfg["urdf_resolved"].endswith("r.urdf")


def test_missing_key_raises(tmp_path):
    bad = _good(); del bad["thresholds"]
    with pytest.raises(ValueError) as e:
        load_config(_write(tmp_path, bad))
    assert "thresholds" in str(e.value)


def test_bad_forward_axis_raises(tmp_path):
    bad = _good(); bad["base_forward_axis"] = "north"
    with pytest.raises(ValueError) as e:
        load_config(_write(tmp_path, bad))
    assert "base_forward_axis" in str(e.value)


def test_bad_modality_raises(tmp_path):
    bad = _good(); bad["cameras"]["head"]["modality"] = "thermal"
    with pytest.raises(ValueError) as e:
        load_config(_write(tmp_path, bad))
    assert "modality" in str(e.value)


def test_missing_urdf_raises(tmp_path):
    bad = _good(urdf_name="nope.urdf")
    p = tmp_path / "cfg.json"; p.write_text(json.dumps(bad))  # no urdf file written
    with pytest.raises(FileNotFoundError):
        load_config(str(p))
