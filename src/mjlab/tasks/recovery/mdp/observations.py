from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply_inverse,
  quat_conjugate,
  quat_mul,
  yaw_quat,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def root_local_rot_tan_norm(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Encode torso orientation with yaw removed using tangent/normal vectors."""
  asset: Entity = env.scene[asset_cfg.name]
  root_quat = asset.data.root_link_quat_w
  local_root_quat = quat_mul(quat_conjugate(yaw_quat(root_quat)), root_quat)
  root_rot = matrix_from_quat(local_root_quat)
  tan_vec = root_rot[:, :, 0]
  norm_vec = root_rot[:, :, 2]
  return torch.cat([tan_vec, norm_vec], dim=-1)


def key_body_pos_b(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Return selected body positions in the root frame."""
  asset: Entity = env.scene[asset_cfg.name]
  body_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids]
  root_pos_w = asset.data.root_link_pos_w.unsqueeze(1)
  root_quat_w = asset.data.root_link_quat_w.unsqueeze(1).expand_as(
    torch.zeros(body_pos_w.shape[0], body_pos_w.shape[1], 4, device=env.device)
  )
  rel_pos_w = body_pos_w - root_pos_w
  rel_pos_b = quat_apply_inverse(root_quat_w, rel_pos_w)
  return rel_pos_b.reshape(rel_pos_b.shape[0], -1)


def root_height(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Root height relative to the local environment origin."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_pos_w[:, 2:3] - env.scene.env_origins[:, 2:3]
