from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _root_height(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]


def _uprightness(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return -asset.data.projected_gravity_b[:, 2]


def _height_gate(
  env: ManagerBasedRlEnv,
  activation_height: float,
  full_weight_height: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Ramp rewards on only when the robot is nearing a standing pose."""
  height = _root_height(env, asset_cfg=asset_cfg)
  denom = max(full_weight_height - activation_height, 1e-6)
  return torch.clamp((height - activation_height) / denom, min=0.0, max=1.0)


def base_height_progress(
  env: ManagerBasedRlEnv,
  minimum_height: float,
  target_height: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Dense shaping term that rewards the torso for rising off the floor."""
  height = _root_height(env, asset_cfg=asset_cfg)
  progress = (height - minimum_height) / max(target_height - minimum_height, 1e-6)
  return torch.clamp(progress, min=0.0, max=1.0)


def target_orientation(
  env: ManagerBasedRlEnv,
  target_base_height_phase3: float,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  base_height = _root_height(env, asset_cfg=asset_cfg)
  standing = base_height > target_base_height_phase3
  xy_sq = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
  return torch.exp(-xy_sq / std**2) * standing.float()


def target_base_height(
  env: ManagerBasedRlEnv,
  base_height_target: float,
  target_base_height_phase3: float,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  base_height = _root_height(env, asset_cfg=asset_cfg)
  standing = base_height > target_base_height_phase3
  error = torch.square((base_height - base_height_target) / std)
  return torch.exp(-error) * standing.float()


def lin_vel_xy(
  env: ManagerBasedRlEnv,
  target_base_height_phase3: float,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  base_height = _root_height(env, asset_cfg=asset_cfg)
  standing = base_height > target_base_height_phase3
  vel_sq = torch.sum(torch.square(asset.data.root_link_lin_vel_b[:, :2]), dim=1)
  return torch.exp(-vel_sq / std**2) * standing.float()


def ang_vel_xy(
  env: ManagerBasedRlEnv,
  target_base_height_phase3: float,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  base_height = _root_height(env, asset_cfg=asset_cfg)
  standing = base_height > target_base_height_phase3
  vel_sq = torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=1)
  return torch.exp(-vel_sq / std**2) * standing.float()


def stand_bonus(
  env: ManagerBasedRlEnv,
  stand_height: float,
  upright_threshold: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  height = _root_height(env, asset_cfg=asset_cfg)
  uprightness = _uprightness(env, asset_cfg=asset_cfg)
  return ((height > stand_height) & (uprightness > upright_threshold)).float()


def late_phase_orientation(
  env: ManagerBasedRlEnv,
  activation_height: float,
  full_weight_height: float,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Tighten uprightness once the robot is almost standing."""
  asset: Entity = env.scene[asset_cfg.name]
  gate = _height_gate(
    env,
    activation_height=activation_height,
    full_weight_height=full_weight_height,
    asset_cfg=asset_cfg,
  )
  xy_sq = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
  return torch.exp(-xy_sq / std**2) * gate


def late_phase_base_motion(
  env: ManagerBasedRlEnv,
  activation_height: float,
  full_weight_height: float,
  lin_vel_std: float,
  ang_vel_std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward a quiet base once recovery transitions into standing."""
  asset: Entity = env.scene[asset_cfg.name]
  gate = _height_gate(
    env,
    activation_height=activation_height,
    full_weight_height=full_weight_height,
    asset_cfg=asset_cfg,
  )
  lin_vel_sq = torch.sum(torch.square(asset.data.root_link_lin_vel_b[:, :2]), dim=1)
  ang_vel_sq = torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=1)
  lin_reward = torch.exp(-lin_vel_sq / lin_vel_std**2)
  ang_reward = torch.exp(-ang_vel_sq / ang_vel_std**2)
  return 0.5 * (lin_reward + ang_reward) * gate


def late_phase_joint_vel_l2(
  env: ManagerBasedRlEnv,
  activation_height: float,
  full_weight_height: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Penalize selected joints for chattering near the final stand."""
  asset: Entity = env.scene[asset_cfg.name]
  gate = _height_gate(
    env,
    activation_height=activation_height,
    full_weight_height=full_weight_height,
  )
  joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
  return torch.mean(torch.square(joint_vel), dim=1) * gate


def late_phase_posture(
  env: ManagerBasedRlEnv,
  activation_height: float,
  full_weight_height: float,
  std: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Encourage selected joints to settle back to their default standing pose."""
  asset: Entity = env.scene[asset_cfg.name]
  default_joint_pos = asset.data.default_joint_pos
  assert default_joint_pos is not None
  gate = _height_gate(
    env,
    activation_height=activation_height,
    full_weight_height=full_weight_height,
  )
  current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
  desired_joint_pos = default_joint_pos[:, asset_cfg.joint_ids]
  error_squared = torch.square(current_joint_pos - desired_joint_pos)
  return torch.exp(-torch.mean(error_squared / std**2, dim=1)) * gate
