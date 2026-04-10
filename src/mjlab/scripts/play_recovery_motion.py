"""Play converted recovery teacher motions on the Unitree G1 in mjlab.

Examples:
  uv run python -m mjlab.scripts.play_recovery_motion \
    --motion-path /path/to/recovery_motions/g1_amp_get_up \
    --clip fallAndGetUp2_subject2_1200_1370

  uv run python -m mjlab.scripts.play_recovery_motion \
    --motion-path /path/to/recovery_motions/g1_amp_get_up \
    --headless --max-frames 120
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.scene import Scene
from mjlab.sim import Simulation
from mjlab.tasks.recovery.config.g1.env_cfgs import unitree_g1_flat_recovery_env_cfg


def _resolve_motion_file(motion_path: str, clip: str | None) -> Path:
    path = Path(motion_path).expanduser()
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Motion path does not exist: {path}")

    files = sorted(path.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in: {path}")

    if clip is None:
        return files[0]

    direct = path / clip
    if direct.suffix == ".npz" and direct.exists():
        return direct

    stem_match = path / f"{clip}.npz"
    if stem_match.exists():
        return stem_match

    available = ", ".join(f.stem for f in files[:10])
    raise FileNotFoundError(
        f"Clip '{clip}' not found in {path}. Example available clips: {available}"
    )


def _load_motion(path: Path) -> tuple[dict[str, torch.Tensor], float]:
    with np.load(path, allow_pickle=False) as data:
        motion = {
            "root_pos": torch.tensor(data["root_pos"], dtype=torch.float32),
            "root_quat": torch.tensor(data["root_quat"], dtype=torch.float32),
            "root_lin_vel": torch.tensor(data["root_lin_vel"], dtype=torch.float32),
            "root_ang_vel": torch.tensor(data["root_ang_vel"], dtype=torch.float32),
            "joint_pos": torch.tensor(data["joint_pos"], dtype=torch.float32),
            "joint_vel": torch.tensor(data["joint_vel"], dtype=torch.float32),
        }
        fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    return motion, fps


def _sync_env0_to_mjdata(sim: Simulation) -> None:
    """Copy environment 0 from batched warp tensors into CPU mjData for rendering."""
    sim_data = sim.data
    mj_data = sim.mj_data
    mj_model = sim.mj_model
    if mj_model.nq > 0:
        mj_data.qpos[:] = sim_data.qpos[0].cpu().numpy()
        mj_data.qvel[:] = sim_data.qvel[0].cpu().numpy()
    if mj_model.nmocap > 0:
        mj_data.mocap_pos[:] = sim_data.mocap_pos[0].cpu().numpy()
        mj_data.mocap_quat[:] = sim_data.mocap_quat[0].cpu().numpy()
    mj_data.xfrc_applied[:] = sim_data.xfrc_applied[0].cpu().numpy()
    mujoco.mj_forward(mj_model, mj_data)


def _apply_frame(
    robot: Entity,
    scene: Scene,
    frame_idx: int,
    motion: dict[str, torch.Tensor],
) -> None:
    device = scene.device
    root_state = robot.data.default_root_state.clone()
    root_state[:, 0:3] = motion["root_pos"][frame_idx:frame_idx + 1].to(device)
    root_state[:, 0:3] += scene.env_origins[:, 0:3]
    root_state[:, 3:7] = motion["root_quat"][frame_idx:frame_idx + 1].to(device)
    root_state[:, 7:10] = motion["root_lin_vel"][frame_idx:frame_idx + 1].to(device)
    root_state[:, 10:13] = motion["root_ang_vel"][frame_idx:frame_idx + 1].to(device)
    robot.write_root_state_to_sim(root_state)

    joint_pos = motion["joint_pos"][frame_idx:frame_idx + 1].to(device)
    joint_vel = motion["joint_vel"][frame_idx:frame_idx + 1].to(device)
    if joint_pos.shape[-1] != robot.num_joints:
        raise ValueError(
            f"Motion joint dim {joint_pos.shape[-1]} does not match robot joints "
            f"{robot.num_joints}."
        )
    robot.write_joint_state_to_sim(joint_pos, joint_vel)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--motion-path",
        required=True,
        help="Path to a converted .npz clip or a directory of converted clips.",
    )
    parser.add_argument(
        "--clip",
        default=None,
        help="Optional clip stem or filename when --motion-path is a directory.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Simulation device, e.g. cpu or cuda:0.",
    )
    parser.add_argument(
        "--loop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Loop playback after the last frame.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run the playback loop without opening a viewer window.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional maximum number of frames to play before exiting.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override playback FPS. Defaults to the stored clip FPS.",
    )
    args = parser.parse_args()

    motion_file = _resolve_motion_file(args.motion_path, args.clip)
    motion, stored_fps = _load_motion(motion_file)
    playback_fps = args.fps or stored_fps

    cfg = unitree_g1_flat_recovery_env_cfg(play=True)
    cfg.scene.num_envs = 1
    scene = Scene(cfg.scene, device=args.device)
    sim = Simulation(
        num_envs=scene.num_envs,
        cfg=cfg.sim,
        model=scene.compile(),
        device=args.device,
    )
    scene.initialize(sim.mj_model, sim.model, sim.data)
    robot: Entity = scene["robot"]
    scene.reset()

    if not args.headless:
        viewer = mujoco.viewer.launch_passive(
            sim.mj_model,
            sim.mj_data,
            show_left_ui=False,
            show_right_ui=False,
        )
        body_id_list, _ = robot.find_bodies(cfg.viewer.body_name)
        if not body_id_list:
            raise ValueError(
                f"Body '{cfg.viewer.body_name}' not found on recovery robot."
            )
        body_id = robot.indexing.bodies[body_id_list[0]].id
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING.value
        viewer.cam.trackbodyid = body_id
        viewer.cam.fixedcamid = -1
        viewer.cam.elevation = cfg.viewer.elevation
        viewer.cam.azimuth = cfg.viewer.azimuth
        viewer.cam.distance = cfg.viewer.distance
    else:
        viewer = None

    print(f"[INFO] Playing motion: {motion_file}")
    print(f"[INFO] Clip frames: {motion['joint_pos'].shape[0]}, fps: {playback_fps}")
    frame_dt = 1.0 / playback_fps
    frame_idx = 0
    played_frames = 0

    try:
        while True:
            if viewer is not None and not viewer.is_running():
                break

            tic = time.perf_counter()
            _apply_frame(robot, scene, frame_idx, motion)
            sim.forward()
            scene.update(sim.mj_model.opt.timestep)

            if viewer is not None:
                _sync_env0_to_mjdata(sim)
                viewer.sync()

            played_frames += 1
            if args.max_frames is not None and played_frames >= args.max_frames:
                break

            frame_idx += 1
            if frame_idx >= motion["joint_pos"].shape[0]:
                if args.loop:
                    frame_idx = 0
                else:
                    break

            sleep_time = frame_dt - (time.perf_counter() - tic)
            if sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        if viewer is not None:
            viewer.close()

    print(f"[OK] Playback finished after {played_frames} frames.")


if __name__ == "__main__":
    main()
