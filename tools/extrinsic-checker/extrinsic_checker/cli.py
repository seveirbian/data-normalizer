"""Command-line entry point: orchestrate per-camera checks."""

import argparse
import sys

from .loader import load_config
from .h5read import open_h5, read_frame_values, has_modality
from .kinematics import Kinematics, build_cfg
from .depth_check import run_depth_check
from .projection_check import run_projection_check
from .report import print_report


def build_parser():
    p = argparse.ArgumentParser(
        prog="extrinsic-check",
        description="Verify dataset camera extrinsics via base-frame reconstruction.",
    )
    p.add_argument("--config", required=True, help="Path to the checker config JSON")
    p.add_argument("--input", required=True, help="Path to the .h5 recording")
    p.add_argument("--camera", nargs="+", default=None,
                   help="Cameras to check (default: all in config)")
    p.add_argument("--frame", type=int, default=0, help="Frame index (default 0)")
    p.add_argument("--out-dir", default="extrinsic_check_out", help="Artifact output dir")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config)
        kin = Kinematics(cfg["urdf_resolved"], cfg["base_link"])
        cameras = args.camera or list(cfg["cameras"].keys())
        with open_h5(args.input) as h5:
            frame_values = read_frame_values(h5, cfg["joint_mapping"], args.frame)
            jcfg = build_cfg(frame_values, cfg["joint_mapping"], kin.robot.actuated_joint_names)
            verdicts = []
            for cam in cameras:
                if cam not in cfg["cameras"]:
                    print(f"error: camera {cam!r} not in config; available {list(cfg['cameras'])}",
                          file=sys.stderr)
                    return 2
                cc = cfg["cameras"][cam]
                modality = cc["modality"]
                if not has_modality(h5, cam, modality):
                    print(f"error: camera {cam!r} modality {modality!r} absent in h5", file=sys.stderr)
                    return 2
                if modality == "depth":
                    v = run_depth_check(h5, cam, cc["mount_link"], kin, jcfg,
                                        cfg["thresholds"], cfg["base_forward_axis"],
                                        args.frame, args.out_dir)
                else:
                    v = run_projection_check(h5, cam, cc["mount_link"],
                                             cc.get("projection_targets", []), kin, jcfg,
                                             args.frame, args.out_dir)
                verdicts.append(v)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    ok = print_report(verdicts)
    return 0 if ok else 1
