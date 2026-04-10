"""Convert amp_rec motion clips from .pkl into mjlab recovery-reset .npz files.

The output format is intentionally minimal and is consumed by
`mjlab.tasks.recovery.mdp.events.RecoveryMotionLoader`.

Usage:
  python -m mjlab.scripts.amp_pkl_to_npz \
    --input /path/to/amp/get_up \
    --output /path/to/mjlab_recovery_npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

AMP_REC_G1_LAB_DOF_NAMES = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)

MJLAB_G1_DOF_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)


def _forward_diff(values: np.ndarray, dt: float) -> np.ndarray:
    vel = np.zeros_like(values, dtype=np.float32)
    if len(values) < 2:
        return vel
    vel[:-1] = (values[1:] - values[:-1]) / dt
    vel[-1] = vel[-2]
    return vel


def _quat_conjugate(quat: np.ndarray) -> np.ndarray:
    out = quat.copy()
    out[..., 1:] *= -1.0
    return out


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = np.moveaxis(q1, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(q2, -1, 0)
    return np.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        axis=-1,
    )


def _quat_ang_vel(quat_wxyz: np.ndarray, dt: float) -> np.ndarray:
    ang_vel = np.zeros((quat_wxyz.shape[0], 3), dtype=np.float32)
    if quat_wxyz.shape[0] < 2:
        return ang_vel

    delta = _quat_mul(quat_wxyz[1:], _quat_conjugate(quat_wxyz[:-1]))
    delta = delta / np.linalg.norm(delta, axis=-1, keepdims=True).clip(min=1e-8)
    neg_w = delta[:, 0] < 0.0
    delta[neg_w] *= -1.0

    w = np.clip(delta[:, 0], -1.0, 1.0)
    xyz = delta[:, 1:]
    xyz_norm = np.linalg.norm(xyz, axis=-1, keepdims=True)
    axis = xyz / np.clip(xyz_norm, 1e-8, None)
    angle = 2.0 * np.arctan2(xyz_norm.squeeze(-1), w)
    ang_vel[:-1] = axis * (angle[:, None] / dt)
    ang_vel[-1] = ang_vel[-2]
    return ang_vel


def _load_motion(path: Path) -> dict:
    try:
        import joblib
    except ImportError as exc:  # pragma: no cover - import error path is trivial
        raise ImportError(
            "joblib is required to read amp_rec .pkl motion files. "
            "Install it in the environment running this converter."
        ) from exc

    motion = joblib.load(path)
    if not isinstance(motion, dict):
        raise TypeError(f"Motion file '{path}' did not contain a dictionary.")
    return motion


def _reorder_g1_joints_from_amp_rec_lab_to_mjlab(
    joint_pos: np.ndarray,
) -> np.ndarray:
    """Reorder AMP recovery clips into mjlab's G1 joint order.

    The source `.pkl` motions were exported with the `lab_dof_names` order from
    `amp_rec/scripts/tools/retarget/config/g1_29dof.yaml`, which differs from
    the Unitree G1 joint order used by `mjlab`.
    """
    if joint_pos.shape[-1] != len(AMP_REC_G1_LAB_DOF_NAMES):
        return joint_pos

    source_index = {name: idx for idx, name in enumerate(AMP_REC_G1_LAB_DOF_NAMES)}
    reorder = [source_index[name] for name in MJLAB_G1_DOF_NAMES]
    return joint_pos[:, reorder]


def convert_motion_file(input_path: Path, output_path: Path) -> None:
    motion = _load_motion(input_path)
    fps = float(motion["fps"])
    dt = 1.0 / fps

    root_pos = np.asarray(motion["root_pos"], dtype=np.float32)
    root_quat = np.asarray(
        motion.get("root_rot", motion.get("root_quat")),
        dtype=np.float32,
    )
    joint_pos = np.asarray(motion["dof_pos"], dtype=np.float32)
    joint_pos = _reorder_g1_joints_from_amp_rec_lab_to_mjlab(joint_pos)

    root_lin_vel = np.asarray(
        motion.get("root_vel_w", _forward_diff(root_pos, dt)),
        dtype=np.float32,
    )
    root_ang_vel = np.asarray(
        motion.get("root_ang_vel_w", _quat_ang_vel(root_quat, dt)),
        dtype=np.float32,
    )
    joint_vel = np.asarray(
        motion.get("dof_vel", _forward_diff(joint_pos, dt)),
        dtype=np.float32,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        fps=np.array([fps], dtype=np.float32),
        root_pos=root_pos,
        root_quat=root_quat,
        root_lin_vel=root_lin_vel,
        root_ang_vel=root_ang_vel,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        joint_names=np.asarray(MJLAB_G1_DOF_NAMES),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help="Input .pkl file or directory containing .pkl motion clips.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for converted .npz clips.",
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser()
    output_dir = Path(args.output).expanduser()
    if input_path.is_dir():
        files = sorted(input_path.glob("*.pkl"))
    elif input_path.suffix == ".pkl":
        files = [input_path]
    else:
        raise ValueError(
            "Input must be a .pkl file or a directory of .pkl files."
        )

    if not files:
        raise ValueError(f"No .pkl files found in '{input_path}'.")

    for file_path in files:
        out_path = output_dir / f"{file_path.stem}.npz"
        convert_motion_file(file_path, out_path)
        print(f"[OK] {file_path.name} -> {out_path}")


if __name__ == "__main__":
    main()
