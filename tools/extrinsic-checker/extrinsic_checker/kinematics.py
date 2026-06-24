"""Joint-config assembly (pure) and a thin yourdfpy FK wrapper."""

import yourdfpy


def build_cfg(frame_values, joint_mapping, actuated_joints):
    """Assemble a {urdf_joint: value} dict.

    frame_values: {group_name: 1D array already indexed at the frame}.
    Applies each entry's `sign` (default 1); unmapped joints default to 0.0.
    """
    cfg = {j: 0.0 for j in actuated_joints}
    for group, spec in joint_mapping.items():
        arr = frame_values[group]
        for e in spec["entries"]:
            idx = e["h5_index"]
            if idx >= len(arr):
                raise ValueError(
                    f"group {group!r}: h5_index {idx} out of range (array width {len(arr)})"
                )
            cfg[e["urdf_joint"]] = float(arr[idx]) * e.get("sign", 1)
    return cfg


class Kinematics:
    def __init__(self, urdf_path, base_link):
        self.robot = yourdfpy.URDF.load(
            urdf_path, load_meshes=False, build_collision_scene_graph=False
        )
        self.base_link = base_link

    def link_transform(self, cfg, link):
        """Return T_base_link (4x4) for `link` expressed in base_link, given cfg."""
        self.robot.update_cfg(cfg)
        return self.robot.get_transform(link, self.base_link)
