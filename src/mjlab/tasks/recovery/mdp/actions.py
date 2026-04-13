from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class RecoveryJointPositionActionCfg(JointPositionActionCfg):
  """Joint-position action with an HoST-style unactuated warmup window."""

  unactuated_steps: int = 30

  def build(self, env: ManagerBasedRlEnv) -> RecoveryJointPositionAction:
    return RecoveryJointPositionAction(self, env)


class RecoveryJointPositionAction(JointPositionAction):
  """Hold current pose for the first few env steps after reset."""

  cfg: RecoveryJointPositionActionCfg

  def apply_actions(self) -> None:
    encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
    target = self._processed_actions - encoder_bias
    warmup_mask = self._env.episode_length_buf < self.cfg.unactuated_steps
    if torch.any(warmup_mask):
      current_joint_pos = self._entity.data.joint_pos[:, self._target_ids]
      target = target.clone()
      target[warmup_mask] = current_joint_pos[warmup_mask]
    self._entity.set_joint_position_target(target, joint_ids=self._target_ids)
