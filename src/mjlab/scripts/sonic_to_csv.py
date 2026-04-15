from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import tyro

import mjlab


def _quat_xyzw_from_ypr(
    yaw: np.ndarray, pitch: np.ndarray, roll: np.ndarray
) -> np.ndarray:
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp
    return np.stack([qx, qy, qz, qw], axis=-1)


def _resolve_pose_layout(
    data: np.ndarray, pose_layout: Literal["auto", "ypr", "quat"]
) -> Literal["ypr", "quat"]:
    if pose_layout != "auto":
        return pose_layout
    num_columns = data.shape[1]
    if num_columns < 35:
        raise ValueError(
            f"Could not infer SONIC pose layout from {num_columns} columns. "
            "Expected at least 35 columns."
        )
    if num_columns == 35:
        return "ypr"
    quat_norms = np.linalg.norm(data[: min(len(data), 100), 3:7], axis=1)
    quat_norm_error = np.abs(quat_norms - 1.0).mean()
    return "quat" if quat_norm_error < 0.1 else "ypr"


def convert_sonic_txt_to_csv(
    input_file: Path,
    output_file: Path,
    pose_layout: Literal["auto", "ypr", "quat"] = "auto",
    quaternion_order: Literal["xyzw", "wxyz"] = "xyzw",
) -> Path:
    data = np.loadtxt(input_file, dtype=np.float32)
    data = np.atleast_2d(data)
    num_rows, num_columns = data.shape
    resolved_layout = _resolve_pose_layout(data, pose_layout)

    if resolved_layout == "ypr":
        if num_columns < 35:
            raise ValueError(
                f"Expected at least 35 columns for ypr layout, got {num_columns}."
            )
        root_pos = data[:, :3]
        yaw_pitch_roll = data[:, 3:6]
        joints = data[:, 6:35]
        root_quat = _quat_xyzw_from_ypr(
            yaw=yaw_pitch_roll[:, 0],
            pitch=yaw_pitch_roll[:, 1],
            roll=yaw_pitch_roll[:, 2],
        )
    else:
        if num_columns < 36:
            raise ValueError(
                f"Expected at least 36 columns for quaternion layout, got {num_columns}."
            )
        root_pos = data[:, :3]
        root_quat = data[:, 3:7]
        joints = data[:, 7:36]
        if quaternion_order == "wxyz":
            root_quat = root_quat[:, [1, 2, 3, 0]]

    csv_data = np.concatenate([root_pos, root_quat, joints], axis=1)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_file, csv_data, delimiter=",", fmt="%.8f")

    print(f"Input file: {input_file}")
    print(f"Detected layout: {resolved_layout}")
    print(f"Input rows: {num_rows}")
    print(f"Input columns: {num_columns}")
    print(f"Ignored trailing columns: {num_columns - csv_data.shape[1]}")
    print(f"Output CSV: {output_file}")
    print(f"Output shape: {csv_data.shape}")
    print(
        "Output layout: [root_pos(3), root_quat_xyzw(4), joint_pos(29)] -> 36 columns"
    )
    return output_file


def main(
    input_file: str,
    output_file: str,
    pose_layout: Literal["auto", "ypr", "quat"] = "auto",
    quaternion_order: Literal["xyzw", "wxyz"] = "xyzw",
):
    """Convert SONIC G1 text dumps into mjlab-compatible 36-column CSV.

    Args:
        input_file: Path to SONIC text dump.
        output_file: Destination CSV path.
        pose_layout: Whether the pose block uses yaw/pitch/roll or quaternion.
        quaternion_order: Quaternion order for quaternion-layout files.
    """
    convert_sonic_txt_to_csv(
        input_file=Path(input_file).expanduser(),
        output_file=Path(output_file).expanduser(),
        pose_layout=pose_layout,
        quaternion_order=quaternion_order,
    )


if __name__ == "__main__":
    tyro.cli(main, config=mjlab.TYRO_FLAGS)
