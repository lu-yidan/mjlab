"""Preview the fallen-pose reset sources used by G1 recovery.

This script can visualize either:

- the fallback canonical fallen poses configured in code
- motion-derived fallen states sampled from converted get-up clips

This is useful for checking penetration, odd joint configurations, and whether
the reset distribution is physically plausible before launching training.
"""

from __future__ import annotations

import argparse
import time

import mujoco
import mujoco.viewer
import torch

from mjlab.entity import Entity
from mjlab.scene import Scene
from mjlab.sim import Simulation
from mjlab.tasks.recovery.config.g1.env_cfgs import (
    G1_FALLEN_POSES,
    unitree_g1_flat_recovery_env_cfg,
)
from mjlab.tasks.recovery.mdp.events import RecoveryMotionLoader
from mjlab.utils.lab_api.math import quat_from_euler_xyz
from mjlab.utils.string import resolve_expr


def _sync_env0_to_mjdata(sim: Simulation) -> None:
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


def _apply_pose(
    robot: Entity,
    scene: Scene,
    pose_name: str,
    device: str,
) -> None:
    pose_cfg = next(p for p in G1_FALLEN_POSES if p.name == pose_name)
    root_state = robot.data.default_root_state.clone()
    root_state[:, 0] = scene.env_origins[:, 0] + pose_cfg.root_pos[0]
    root_state[:, 1] = scene.env_origins[:, 1] + pose_cfg.root_pos[1]
    root_state[:, 2] = scene.env_origins[:, 2] + pose_cfg.root_pos[2]
    root_state[:, 3:7] = quat_from_euler_xyz(
        torch.tensor([pose_cfg.root_rpy[0]], device=device),
        torch.tensor([pose_cfg.root_rpy[1]], device=device),
        torch.tensor([pose_cfg.root_rpy[2]], device=device),
    )
    root_state[:, 7:13] = 0.0
    robot.write_root_state_to_sim(root_state)

    joint_pos = torch.tensor(
        [resolve_expr(pose_cfg.joint_pos, robot.joint_names, 0.0)],
        dtype=torch.float32,
        device=device,
    )
    joint_vel = torch.tensor(
        [resolve_expr(pose_cfg.joint_vel, robot.joint_names, 0.0)],
        dtype=torch.float32,
        device=device,
    )
    robot.write_joint_state_to_sim(joint_pos, joint_vel)


def _apply_motion_pose(
    robot: Entity,
    scene: Scene,
    sample: dict[str, torch.Tensor],
    sample_idx: int,
    height_offset: float,
    zero_velocity: bool,
) -> None:
    device = scene.device
    root_state = robot.data.default_root_state.clone()
    root_state[:, 0:2] = scene.env_origins[:, 0:2]
    root_state[:, 2] = (
        sample["root_pos"][sample_idx:sample_idx + 1, 2]
        + scene.env_origins[:, 2]
        + height_offset
    )
    root_state[:, 3:7] = sample["root_quat"][sample_idx:sample_idx + 1].to(device)
    if zero_velocity:
        root_state[:, 7:13] = 0.0
    else:
        root_state[:, 7:10] = sample["root_lin_vel"][sample_idx:sample_idx + 1].to(
            device
        )
        root_state[:, 10:13] = sample["root_ang_vel"][sample_idx:sample_idx + 1].to(
            device
        )
    robot.write_root_state_to_sim(root_state)

    joint_pos = sample["joint_pos"][sample_idx:sample_idx + 1].to(device)
    if zero_velocity:
        joint_vel = torch.zeros_like(joint_pos)
    else:
        joint_vel = sample["joint_vel"][sample_idx:sample_idx + 1].to(device)
    robot.write_joint_state_to_sim(joint_pos, joint_vel)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--motion-path",
        default=None,
        help="Optional converted motion directory/file for motion-derived fallen states.",
    )
    parser.add_argument(
        "--pose",
        default=None,
        choices=[p.name for p in G1_FALLEN_POSES],
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=4,
        help="How many motion-derived fallen states to preview.",
    )
    parser.add_argument(
        "--progress-max",
        type=float,
        default=0.12,
        help="Maximum normalized motion progress for fallen-state sampling.",
    )
    parser.add_argument(
        "--height-offset",
        type=float,
        default=0.05,
        help="Extra root height offset when previewing motion-derived states.",
    )
    parser.add_argument(
        "--zero-velocity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to zero velocities while previewing motion-derived states.",
    )
    parser.add_argument(
        "--seconds-per-pose",
        type=float,
        default=3.0,
        help="How long to hold each pose before moving to the next one.",
    )
    parser.add_argument(
        "--settle-steps",
        type=int,
        default=0,
        help="Optional number of zero-action physics steps after placing each pose.",
    )
    parser.add_argument(
        "--show-settle-steps",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Render the intermediate settle steps instead of only the final pose.",
    )
    parser.add_argument(
        "--settle-step-delay",
        type=float,
        default=0.03,
        help="Delay in seconds between rendered settle steps.",
    )
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

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
            raise ValueError(f"Body '{cfg.viewer.body_name}' not found.")
        body_id = robot.indexing.bodies[body_id_list[0]].id
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING.value
        viewer.cam.trackbodyid = body_id
        viewer.cam.fixedcamid = -1
        viewer.cam.elevation = cfg.viewer.elevation
        viewer.cam.azimuth = cfg.viewer.azimuth
        viewer.cam.distance = cfg.viewer.distance
    else:
        viewer = None

    motion_loader = None
    motion_samples = None
    if args.motion_path is not None:
        motion_loader = RecoveryMotionLoader(
            motion_path=args.motion_path,
            motion_weights={},
            device=args.device,
        )
        motion_samples = motion_loader.sample(
            num_samples=args.num_samples,
            min_progress=0.0,
            max_progress=args.progress_max,
        )

    pose_names = None
    if motion_loader is None:
        pose_names = (
            [args.pose]
            if args.pose is not None
            else [p.name for p in G1_FALLEN_POSES]
        )
    try:
        if motion_loader is not None and motion_samples is not None:
            for sample_idx in range(args.num_samples):
                print(f"[INFO] Previewing motion-derived fallen sample: {sample_idx}")
                _apply_motion_pose(
                    robot,
                    scene,
                    motion_samples,
                    sample_idx,
                    height_offset=args.height_offset,
                    zero_velocity=args.zero_velocity,
                )
                sim.forward()
                scene.update(sim.mj_model.opt.timestep)
                for _ in range(args.settle_steps):
                    scene.write_data_to_sim()
                    sim.step()
                    scene.update(sim.mj_model.opt.timestep)
                    sim.forward()
                    if viewer is not None and args.show_settle_steps:
                        if not viewer.is_running():
                            break
                        _sync_env0_to_mjdata(sim)
                        viewer.sync()
                        time.sleep(args.settle_step_delay)
                if viewer is not None:
                    start = time.time()
                    while (
                        viewer.is_running()
                        and time.time() - start < args.seconds_per_pose
                    ):
                        _sync_env0_to_mjdata(sim)
                        viewer.sync()
                        time.sleep(0.02)
                else:
                    root_h = robot.data.root_link_pos_w[:, 2] - scene.env_origins[:, 2]
                    print("[INFO] root height:", float(root_h[0]))
        else:
            assert pose_names is not None
            for pose_name in pose_names:
                print(f"[INFO] Previewing fallen pose: {pose_name}")
                _apply_pose(robot, scene, pose_name, args.device)
                sim.forward()
                scene.update(sim.mj_model.opt.timestep)
                for _ in range(args.settle_steps):
                    scene.write_data_to_sim()
                    sim.step()
                    scene.update(sim.mj_model.opt.timestep)
                    sim.forward()
                    if viewer is not None and args.show_settle_steps:
                        if not viewer.is_running():
                            break
                        _sync_env0_to_mjdata(sim)
                        viewer.sync()
                        time.sleep(args.settle_step_delay)
                if viewer is not None:
                    start = time.time()
                    while (
                        viewer.is_running()
                        and time.time() - start < args.seconds_per_pose
                    ):
                        _sync_env0_to_mjdata(sim)
                        viewer.sync()
                        time.sleep(0.02)
                else:
                    root_h = robot.data.root_link_pos_w[:, 2] - scene.env_origins[:, 2]
                    print("[INFO] root height:", float(root_h[0]))
    finally:
        if viewer is not None:
            viewer.close()


if __name__ == "__main__":
    main()
