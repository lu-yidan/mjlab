from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import torch

from mjlab.envs.mdp.rewards import action_acc_l2
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.recovery.mdp.rewards import (
  late_phase_base_motion,
  late_phase_joint_vel_l2,
  late_phase_orientation,
  late_phase_posture,
)


class _MockScene(dict):
  env_origins: torch.Tensor


def _make_env() -> Mock:
  env = Mock()
  env.num_envs = 2
  env.device = "cpu"
  env.action_manager = SimpleNamespace(
    action=torch.tensor([[0.4, -0.2, 0.1], [0.3, 0.1, -0.2]], dtype=torch.float32),
    prev_action=torch.tensor(
      [[0.2, -0.1, 0.05], [0.1, 0.1, -0.2]], dtype=torch.float32
    ),
    prev_prev_action=torch.tensor(
      [[0.0, 0.0, 0.0], [0.0, 0.1, -0.2]], dtype=torch.float32
    ),
  )

  robot = Mock()
  robot.data = SimpleNamespace(
    root_link_pos_w=torch.tensor(
      [[0.0, 0.0, 0.62], [0.0, 0.0, 0.74]], dtype=torch.float32
    ),
    projected_gravity_b=torch.tensor(
      [[0.1, 0.0, -0.99], [0.02, 0.01, -0.999]], dtype=torch.float32
    ),
    root_link_lin_vel_b=torch.tensor(
      [[0.4, 0.0, 0.0], [0.02, 0.01, 0.0]], dtype=torch.float32
    ),
    root_link_ang_vel_b=torch.tensor(
      [[0.6, 0.0, 0.0], [0.03, 0.02, 0.0]], dtype=torch.float32
    ),
    joint_vel=torch.tensor([[4.0, 5.0, 1.0], [0.3, 0.2, 2.5]], dtype=torch.float32),
    joint_pos=torch.tensor([[0.0, 0.0, 1.5], [0.25, -0.2, 1.5]], dtype=torch.float32),
    default_joint_pos=torch.tensor(
      [[0.0, 0.0, 1.5], [0.0, 0.0, 1.5]], dtype=torch.float32
    ),
  )

  scene = _MockScene(robot=robot)
  scene.env_origins = torch.zeros((2, 3), dtype=torch.float32)
  env.scene = scene
  return env


def test_late_phase_rewards_turn_on_near_standing() -> None:
  env = _make_env()

  orientation = late_phase_orientation(
    env,
    activation_height=0.68,
    full_weight_height=0.75,
    std=0.2,
  )
  base_motion = late_phase_base_motion(
    env,
    activation_height=0.68,
    full_weight_height=0.75,
    lin_vel_std=0.15,
    ang_vel_std=0.2,
  )

  assert orientation[0].item() == 0.0
  assert base_motion[0].item() == 0.0
  assert orientation[1].item() > 0.0
  assert base_motion[1].item() > 0.0


def test_late_phase_joint_vel_penalty_targets_selected_joints() -> None:
  env = _make_env()
  asset_cfg = SceneEntityCfg(name="robot", joint_ids=[0, 1])

  penalty = late_phase_joint_vel_l2(
    env,
    activation_height=0.66,
    full_weight_height=0.74,
    asset_cfg=asset_cfg,
  )

  assert penalty[0].item() == 0.0
  assert penalty[1].item() > 0.0
  assert penalty[1].item() < 0.1


def test_late_phase_posture_prefers_default_pose() -> None:
  env = _make_env()
  env.scene["robot"].data.root_link_pos_w[:, 2] = 0.75
  asset_cfg = SceneEntityCfg(name="robot", joint_ids=[0, 1])

  posture = late_phase_posture(
    env,
    activation_height=0.68,
    full_weight_height=0.75,
    std=0.35,
    asset_cfg=asset_cfg,
  )

  assert posture[0].item() > posture[1].item()


def test_action_acc_l2_penalizes_second_order_action_chatter() -> None:
  env = _make_env()

  penalty = action_acc_l2(env)

  assert penalty[0].item() == 0.0
  assert penalty[1].item() > 0.0
