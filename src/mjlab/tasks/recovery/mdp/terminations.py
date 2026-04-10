from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def joint_vel_out_of_manual_limit(
  env: ManagerBasedRlEnv,
  max_velocity: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Terminate environments that hit obviously invalid joint speeds."""
  asset: Entity = env.scene[asset_cfg.name]
  joint_vel = torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids])
  return torch.any(joint_vel > max_velocity, dim=1)
