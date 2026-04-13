from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul, sample_uniform
from mjlab.utils.string import resolve_expr

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


class _RecoveryMotionClip:
  def __init__(self, path: Path, device: str):
    with np.load(path, allow_pickle=False) as data:
      self.name = path.stem
      self.root_pos = self._tensor(data, "root_pos", "root_pos_w", device=device)
      self.root_quat = self._tensor(data, "root_quat", "root_rot", device=device)
      self.root_lin_vel = self._tensor(
        data, "root_lin_vel", "root_vel_w", device=device
      )
      self.root_ang_vel = self._tensor(
        data, "root_ang_vel", "root_ang_vel_w", device=device
      )
      self.joint_pos = self._tensor(data, "joint_pos", "dof_pos", device=device)
      self.joint_vel = self._tensor(data, "joint_vel", "dof_vel", device=device)

    self.num_frames = int(self.root_pos.shape[0])
    if self.num_frames < 1:
      raise ValueError(f"Motion clip '{path}' has no frames.")

  @staticmethod
  def _tensor(data, primary: str, fallback: str, device: str) -> torch.Tensor:
    key = primary if primary in data else fallback
    if key not in data:
      raise KeyError(f"Expected '{primary}' or '{fallback}' in motion archive.")
    return torch.tensor(data[key], dtype=torch.float32, device=device)


class RecoveryMotionLoader:
  """Load one or more recovery motion clips for teacher-state initialization."""

  def __init__(
    self,
    motion_path: str,
    motion_weights: dict[str, float],
    device: str,
  ) -> None:
    path = Path(motion_path).expanduser()
    if not path.exists():
      raise FileNotFoundError(f"Recovery motion path does not exist: {path}")

    if path.is_dir():
      files = sorted(path.glob("*.npz"))
    elif path.suffix == ".npz":
      files = [path]
    else:
      raise ValueError(
        f"Unsupported recovery motion path '{path}'. Expected .npz file or directory."
      )
    if not files:
      raise ValueError(f"No .npz motion files found under '{path}'.")

    clips = [_RecoveryMotionClip(file_path, device=device) for file_path in files]
    if motion_weights:
      filtered: list[_RecoveryMotionClip] = []
      weights = []
      for clip in clips:
        weight = motion_weights.get(clip.name, 0.0)
        if weight > 0.0:
          filtered.append(clip)
          weights.append(weight)
      if not filtered:
        raise ValueError(
          "Recovery motion weights did not match any .npz clips with positive weight."
        )
      clips = filtered
      weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    else:
      weight_tensor = torch.ones(len(clips), dtype=torch.float32, device=device)

    self._clips = clips
    self._weights = weight_tensor / weight_tensor.sum()
    self.device = device
    self.num_joints = int(self._clips[0].joint_pos.shape[1])

  def sample(
    self,
    num_samples: int,
    min_progress: float,
    max_progress: float,
  ) -> dict[str, torch.Tensor]:
    if num_samples <= 0:
      raise ValueError("num_samples must be positive.")
    min_progress = float(max(0.0, min(1.0, min_progress)))
    max_progress = float(max(min_progress, min(1.0, max_progress)))

    clip_ids = torch.multinomial(self._weights, num_samples, replacement=True)
    root_pos = torch.zeros((num_samples, 3), device=self.device)
    root_quat = torch.zeros((num_samples, 4), device=self.device)
    root_lin_vel = torch.zeros((num_samples, 3), device=self.device)
    root_ang_vel = torch.zeros((num_samples, 3), device=self.device)
    joint_pos = torch.zeros((num_samples, self.num_joints), device=self.device)
    joint_vel = torch.zeros((num_samples, self.num_joints), device=self.device)

    for clip_idx, clip in enumerate(self._clips):
      mask = clip_ids == clip_idx
      if not torch.any(mask):
        continue
      low = int(round(min_progress * max(clip.num_frames - 1, 0)))
      high = int(round(max_progress * max(clip.num_frames - 1, 0)))
      high = max(low, high)
      frame_ids = torch.randint(low, high + 1, (int(mask.sum().item()),), device=self.device)
      root_pos[mask] = clip.root_pos[frame_ids]
      root_quat[mask] = clip.root_quat[frame_ids]
      root_lin_vel[mask] = clip.root_lin_vel[frame_ids]
      root_ang_vel[mask] = clip.root_ang_vel[frame_ids]
      joint_pos[mask] = clip.joint_pos[frame_ids]
      joint_vel[mask] = clip.joint_vel[frame_ids]

    return {
      "root_pos": root_pos,
      "root_quat": root_quat,
      "root_lin_vel": root_lin_vel,
      "root_ang_vel": root_ang_vel,
      "joint_pos": joint_pos,
      "joint_vel": joint_vel,
    }


class RecoveryReset(ManagerTermBase):
  """Reset into fallen states, optionally mixing in teacher motion states."""

  def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
    super().__init__(env)
    self._cfg = cfg
    self.reference_probability = 0.0
    self._motion_loader: RecoveryMotionLoader | None = None

    teacher_cfg = getattr(env.cfg, "teacher", None)
    if teacher_cfg is not None:
      self.reference_probability = teacher_cfg.reference_probability
      if teacher_cfg.motion_path:
        self._motion_loader = RecoveryMotionLoader(
          motion_path=teacher_cfg.motion_path,
          motion_weights=teacher_cfg.motion_weights,
          device=env.device,
        )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    base_height_range: tuple[float, float] = (0.18, 0.38),
    joint_position_range: tuple[float, float] = (-0.45, 0.45),
    joint_velocity_range: tuple[float, float] = (-1.0, 1.0),
    fallen_pose_library=(),
    fallen_pose_probability: float = 0.4,
    random_fall_probability: float = 0.0,
    motion_fallen_progress_range: tuple[float, float] = (0.0, 0.12),
    motion_fallen_zero_velocity: bool = True,
    fallen_pose_height_offset: float = 0.02,
    fallen_pose_xy_jitter: float = 0.01,
    fallen_pose_yaw_jitter: float = 0.1,
    fallen_pose_joint_jitter: float = 0.03,
    root_velocity_range: dict[str, tuple[float, float]] | None = None,
  ) -> None:
    if env_ids is None:
      env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

    asset: Entity = env.scene[asset_cfg.name]
    teacher_cfg = getattr(env.cfg, "teacher", None)
    teacher_probability = float(max(0.0, min(1.0, self.reference_probability)))

    random_fall_probability = float(max(0.0, min(1.0, random_fall_probability)))
    fallen_pose_probability = float(max(0.0, min(1.0, fallen_pose_probability)))

    teacher_mask = torch.rand(len(env_ids), device=env.device) < teacher_probability
    remaining_mask = ~teacher_mask
    random_mask = torch.zeros(len(env_ids), dtype=torch.bool, device=env.device)
    if random_fall_probability > 0.0 and torch.any(remaining_mask):
      remaining_ids = torch.where(remaining_mask)[0]
      remaining_probability = max(1.0 - teacher_probability, 1e-6)
      random_conditional_probability = min(
        1.0, random_fall_probability / remaining_probability
      )
      random_mask[remaining_ids] = (
        torch.rand(len(remaining_ids), device=env.device)
        < random_conditional_probability
      )
    remaining_after_random_mask = ~(teacher_mask | random_mask)
    fallen_mask = torch.zeros(len(env_ids), dtype=torch.bool, device=env.device)
    if torch.any(remaining_after_random_mask):
      remaining_ids = torch.where(remaining_after_random_mask)[0]
      remaining_probability = max(
        1.0 - teacher_probability - random_fall_probability, 1e-6
      )
      fallen_conditional_probability = min(
        1.0, fallen_pose_probability / remaining_probability
      )
      fallen_mask[remaining_ids] = (
        torch.rand(len(remaining_ids), device=env.device)
        < fallen_conditional_probability
      )
    uncovered_mask = ~(teacher_mask | random_mask | fallen_mask)
    random_mask |= uncovered_mask

    root_pose = asset.data.default_root_state[env_ids, :7].clone()
    root_vel = torch.zeros((len(env_ids), 6), device=env.device)
    joint_pos = asset.data.default_joint_pos[env_ids][:, asset_cfg.joint_ids].clone()
    joint_vel = torch.zeros_like(joint_pos)

    if torch.any(fallen_mask):
      fallen_ids = env_ids[fallen_mask]
      pose_src = self._sample_fallen_pose_state(
        env=env,
        env_ids=fallen_ids,
        asset=asset,
        asset_cfg=asset_cfg,
        fallen_pose_library=fallen_pose_library,
        motion_fallen_progress_range=motion_fallen_progress_range,
        motion_fallen_zero_velocity=motion_fallen_zero_velocity,
        height_offset=fallen_pose_height_offset,
        xy_jitter=fallen_pose_xy_jitter,
        yaw_jitter=fallen_pose_yaw_jitter,
        joint_jitter=fallen_pose_joint_jitter,
      )
      root_pose[fallen_mask] = pose_src["root_pose"]
      root_vel[fallen_mask] = pose_src["root_vel"]
      joint_pos[fallen_mask] = pose_src["joint_pos"]
      joint_vel[fallen_mask] = pose_src["joint_vel"]

    if torch.any(random_mask):
      random_ids = env_ids[random_mask]
      rand_pose, rand_vel, rand_joint_pos, rand_joint_vel = self._sample_random_fall_state(
        env=env,
        env_ids=random_ids,
        asset=asset,
        asset_cfg=asset_cfg,
        base_height_range=base_height_range,
        joint_position_range=joint_position_range,
        joint_velocity_range=joint_velocity_range,
        root_velocity_range=root_velocity_range or {},
      )
      root_pose[random_mask] = rand_pose
      root_vel[random_mask] = rand_vel
      joint_pos[random_mask] = rand_joint_pos
      joint_vel[random_mask] = rand_joint_vel

    if (
      teacher_cfg is not None
      and self._motion_loader is not None
      and teacher_probability > 0.0
      and torch.any(teacher_mask)
    ):
      ref_ids = env_ids[teacher_mask]
      sample = self._motion_loader.sample(
        num_samples=len(ref_ids),
        min_progress=teacher_cfg.min_progress,
        max_progress=teacher_cfg.max_progress,
      )
      root_pose[teacher_mask, 0:2] = env.scene.env_origins[ref_ids, 0:2]
      root_pose[teacher_mask, 2] = (
        sample["root_pos"][:, 2]
        + env.scene.env_origins[ref_ids, 2]
        + teacher_cfg.height_offset
      )
      root_pose[teacher_mask, 3:7] = sample["root_quat"]
      root_vel[teacher_mask, 0:3] = sample["root_lin_vel"]
      root_vel[teacher_mask, 3:6] = sample["root_ang_vel"]
      joint_pos[teacher_mask] = sample["joint_pos"]
      joint_vel[teacher_mask] = sample["joint_vel"]

    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, slice):
      joint_ids = torch.arange(asset.num_joints, device=env.device)
    elif isinstance(joint_ids, list):
      joint_ids = torch.tensor(joint_ids, device=env.device)

    asset.write_root_link_pose_to_sim(root_pose, env_ids=env_ids)
    asset.write_root_link_velocity_to_sim(root_vel, env_ids=env_ids)
    asset.write_joint_state_to_sim(
      joint_pos,
      joint_vel,
      env_ids=env_ids,
      joint_ids=joint_ids,
    )

  def _sample_fallen_pose_state(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset: Entity,
    asset_cfg: SceneEntityCfg,
    fallen_pose_library,
    motion_fallen_progress_range: tuple[float, float],
    motion_fallen_zero_velocity: bool,
    height_offset: float,
    xy_jitter: float,
    yaw_jitter: float,
    joint_jitter: float,
  ) -> dict[str, torch.Tensor]:
    if self._motion_loader is not None:
      return self._sample_motion_fallen_pose_state(
        env=env,
        env_ids=env_ids,
        height_offset=height_offset,
        progress_range=motion_fallen_progress_range,
        zero_velocity=motion_fallen_zero_velocity,
      )
    if fallen_pose_library:
      return self._sample_explicit_fallen_pose_state(
        env=env,
        env_ids=env_ids,
        asset=asset,
        asset_cfg=asset_cfg,
        fallen_pose_library=fallen_pose_library,
        height_offset=height_offset,
        xy_jitter=xy_jitter,
        yaw_jitter=yaw_jitter,
        joint_jitter=joint_jitter,
      )
    rand_pose, rand_vel, rand_joint_pos, rand_joint_vel = self._sample_random_fall_state(
      env=env,
      env_ids=env_ids,
      asset=asset,
      asset_cfg=asset_cfg,
      base_height_range=(0.22, 0.28),
      joint_position_range=(-0.1, 0.1),
      joint_velocity_range=(0.0, 0.0),
      root_velocity_range={},
    )
    return {
      "root_pose": rand_pose,
      "root_vel": rand_vel,
      "joint_pos": rand_joint_pos,
      "joint_vel": rand_joint_vel,
    }

  def _sample_motion_fallen_pose_state(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    height_offset: float,
    progress_range: tuple[float, float],
    zero_velocity: bool,
  ) -> dict[str, torch.Tensor]:
    assert self._motion_loader is not None
    sample = self._motion_loader.sample(
      num_samples=len(env_ids),
      min_progress=progress_range[0],
      max_progress=progress_range[1],
    )
    root_pose = torch.zeros((len(env_ids), 7), device=env.device)
    root_pose[:, 0:2] = env.scene.env_origins[env_ids, 0:2]
    root_pose[:, 2] = sample["root_pos"][:, 2] + env.scene.env_origins[env_ids, 2] + height_offset
    root_pose[:, 3:7] = sample["root_quat"]
    if zero_velocity:
      root_vel = torch.zeros((len(env_ids), 6), device=env.device)
      joint_vel = torch.zeros_like(sample["joint_pos"])
    else:
      root_vel = torch.cat([sample["root_lin_vel"], sample["root_ang_vel"]], dim=-1)
      joint_vel = sample["joint_vel"]
    return {
      "root_pose": root_pose,
      "root_vel": root_vel,
      "joint_pos": sample["joint_pos"],
      "joint_vel": joint_vel,
    }

  def _sample_explicit_fallen_pose_state(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset: Entity,
    asset_cfg: SceneEntityCfg,
    fallen_pose_library,
    height_offset: float,
    xy_jitter: float,
    yaw_jitter: float,
    joint_jitter: float,
  ) -> dict[str, torch.Tensor]:
    num_envs = len(env_ids)
    root_pose = asset.data.default_root_state[env_ids, :7].clone()
    root_vel = torch.zeros((num_envs, 6), device=env.device)
    joint_pos = asset.data.default_joint_pos[env_ids][:, asset_cfg.joint_ids].clone()
    joint_vel = torch.zeros_like(joint_pos)

    pose_indices = torch.randint(0, len(fallen_pose_library), (num_envs,), device=env.device)
    joint_names = asset.joint_names
    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, slice):
      joint_ids = torch.arange(asset.num_joints, device=env.device)
    elif isinstance(joint_ids, list):
      joint_ids = torch.tensor(joint_ids, device=env.device)

    for pose_idx, pose_cfg in enumerate(fallen_pose_library):
      mask = pose_indices == pose_idx
      if not torch.any(mask):
        continue
      count = int(mask.sum().item())
      masked_env_ids = env_ids[mask]

      base_quat = quat_from_euler_xyz(
        torch.full((count,), pose_cfg.root_rpy[0], device=env.device),
        torch.full((count,), pose_cfg.root_rpy[1], device=env.device),
        torch.full((count,), pose_cfg.root_rpy[2], device=env.device),
      )
      yaw_delta = quat_from_euler_xyz(
        torch.zeros(count, device=env.device),
        torch.zeros(count, device=env.device),
        sample_uniform(
          torch.tensor(-yaw_jitter, device=env.device),
          torch.tensor(yaw_jitter, device=env.device),
          (count,),
          env.device,
        ),
      )
      root_pose[mask, 0] = (
        env.scene.env_origins[masked_env_ids, 0]
        + pose_cfg.root_pos[0]
        + sample_uniform(
          torch.tensor(-xy_jitter, device=env.device),
          torch.tensor(xy_jitter, device=env.device),
          (count,),
          env.device,
        )
      )
      root_pose[mask, 1] = (
        env.scene.env_origins[masked_env_ids, 1]
        + pose_cfg.root_pos[1]
        + sample_uniform(
          torch.tensor(-xy_jitter, device=env.device),
          torch.tensor(xy_jitter, device=env.device),
          (count,),
          env.device,
        )
      )
      root_pose[mask, 2] = (
        env.scene.env_origins[masked_env_ids, 2]
        + pose_cfg.root_pos[2]
        + height_offset
      )
      root_pose[mask, 3:7] = quat_mul(yaw_delta, base_quat)

      pose_joint_pos = torch.tensor(
        resolve_expr(pose_cfg.joint_pos, joint_names, 0.0),
        dtype=torch.float32,
        device=env.device,
      )[joint_ids]
      pose_joint_vel = torch.tensor(
        resolve_expr(pose_cfg.joint_vel, joint_names, 0.0),
        dtype=torch.float32,
        device=env.device,
      )[joint_ids]
      joint_pos[mask] = pose_joint_pos.unsqueeze(0).repeat(count, 1)
      joint_vel[mask] = pose_joint_vel.unsqueeze(0).repeat(count, 1)

      if joint_jitter > 0.0:
        joint_pos[mask] += sample_uniform(
          torch.tensor(-joint_jitter, device=env.device),
          torch.tensor(joint_jitter, device=env.device),
          joint_pos[mask].shape,
          env.device,
        )

    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids][:, joint_ids]
    joint_pos = joint_pos.clamp(
      min=joint_pos_limits[..., 0],
      max=joint_pos_limits[..., 1],
    )

    return {
      "root_pose": root_pose,
      "root_vel": root_vel,
      "joint_pos": joint_pos,
      "joint_vel": joint_vel,
    }

  def _sample_random_fall_state(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset: Entity,
    asset_cfg: SceneEntityCfg,
    base_height_range: tuple[float, float],
    joint_position_range: tuple[float, float],
    joint_velocity_range: tuple[float, float],
    root_velocity_range: dict[str, tuple[float, float]],
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_envs = len(env_ids)
    default_root_state = asset.data.default_root_state
    default_joint_pos = asset.data.default_joint_pos
    default_joint_vel = asset.data.default_joint_vel
    soft_joint_pos_limits = asset.data.soft_joint_pos_limits
    assert default_root_state is not None
    assert default_joint_pos is not None
    assert default_joint_vel is not None
    assert soft_joint_pos_limits is not None

    root_pose = default_root_state[env_ids, :7].clone()
    root_pose[:, 0:2] = env.scene.env_origins[env_ids, 0:2]
    root_pose[:, 2] = (
      sample_uniform(
        torch.tensor(base_height_range[0], device=env.device),
        torch.tensor(base_height_range[1], device=env.device),
        (num_envs, 1),
        env.device,
      ).squeeze(-1)
      + env.scene.env_origins[env_ids, 2]
    )

    roll, pitch = self._sample_fall_orientation(num_envs=num_envs, device=env.device)
    yaw = sample_uniform(
      torch.tensor(-math.pi, device=env.device),
      torch.tensor(math.pi, device=env.device),
      (num_envs,),
      env.device,
    )
    root_pose[:, 3:7] = quat_from_euler_xyz(roll, pitch, yaw)

    range_list = [
      root_velocity_range.get(key, (0.0, 0.0))
      for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    root_ranges = torch.tensor(range_list, device=env.device)
    root_vel = sample_uniform(
      root_ranges[:, 0],
      root_ranges[:, 1],
      (num_envs, 6),
      env.device,
    )

    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, slice):
      joint_ids = torch.arange(asset.num_joints, device=env.device)
    elif isinstance(joint_ids, list):
      joint_ids = torch.tensor(joint_ids, device=env.device)

    joint_pos = default_joint_pos[env_ids][:, joint_ids].clone()
    joint_pos += sample_uniform(
      torch.tensor(joint_position_range[0], device=env.device),
      torch.tensor(joint_position_range[1], device=env.device),
      joint_pos.shape,
      env.device,
    )
    joint_pos_limits = soft_joint_pos_limits[env_ids][:, joint_ids]
    joint_pos = joint_pos.clamp(
      min=joint_pos_limits[..., 0],
      max=joint_pos_limits[..., 1],
    )

    joint_vel = default_joint_vel[env_ids][:, joint_ids].clone()
    joint_vel += sample_uniform(
      torch.tensor(joint_velocity_range[0], device=env.device),
      torch.tensor(joint_velocity_range[1], device=env.device),
      joint_vel.shape,
      env.device,
    )

    return root_pose, root_vel, joint_pos, joint_vel

  @staticmethod
  def _sample_fall_orientation(
    num_envs: int,
    device: str,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    mode = torch.randint(0, 5, (num_envs,), device=device)
    jitter = sample_uniform(
      torch.tensor(-0.2, device=device),
      torch.tensor(0.2, device=device),
      (num_envs, 2),
      device,
    )
    roll = torch.zeros(num_envs, device=device)
    pitch = torch.zeros(num_envs, device=device)

    roll = torch.where(mode == 0, math.pi + jitter[:, 0], roll)
    roll = torch.where(mode == 1, 0.5 * math.pi + jitter[:, 0], roll)
    roll = torch.where(mode == 2, -0.5 * math.pi + jitter[:, 0], roll)
    pitch = torch.where(mode == 3, 0.5 * math.pi + jitter[:, 1], pitch)
    pitch = torch.where(mode == 4, -0.5 * math.pi + jitter[:, 1], pitch)

    return roll, pitch
