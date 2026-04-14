from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import torch

from .events import RecoveryReset, UpwardAssistance

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class ProbabilityStage(TypedDict):
  step: int
  probability: float


class PushVelocityStage(TypedDict):
  step: int
  velocity_range: dict[str, tuple[float, float]]


def upward_assistance_force(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice,
  event_name: str,
  success_body_name: str,
  success_height: float,
  success_upright_threshold: float,
  decrement: float,
  min_force: float,
) -> torch.Tensor:
  """Reduce upward assistance for environments that end episodes successfully."""
  if isinstance(env_ids, slice):
    env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]
  event_cfg = env.event_manager.get_term_cfg(event_name)
  event_term = event_cfg.func
  if not isinstance(event_term, UpwardAssistance):
    raise TypeError(
      f"Event '{event_name}' must use UpwardAssistance, got "
      f"{type(event_term).__name__}."
    )

  asset = env.scene["robot"]
  body_ids, _ = asset.find_bodies(success_body_name)
  if not body_ids:
    raise ValueError(f"Body '{success_body_name}' not found on robot.")
  body_height = (
    asset.data.body_link_pos_w[env_ids, body_ids[0], 2]
    - env.scene.env_origins[env_ids, 2]
  )
  uprightness = -asset.data.projected_gravity_b[env_ids, 2]
  success = (body_height > success_height) & (uprightness > success_upright_threshold)
  if torch.any(success):
    success_ids = env_ids[success]
    event_term.force[success_ids] = (
      event_term.force[success_ids] - decrement
    ).clamp(min=min_force)
  return torch.mean(event_term.force)


def reference_reset_probability(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice,
  event_name: str,
  stages: list[ProbabilityStage],
) -> torch.Tensor:
  """Decay teacher-motion resets as recovery training matures."""
  del env_ids  # Unused.
  event_cfg = env.event_manager.get_term_cfg(event_name)
  event_term = event_cfg.func
  if not isinstance(event_term, RecoveryReset):
    raise TypeError(
      f"Event '{event_name}' must use RecoveryReset, got {type(event_term).__name__}."
    )
  for stage in stages:
    if env.common_step_counter >= stage["step"]:
      event_term.reference_probability = stage["probability"]
  return torch.tensor(event_term.reference_probability, device=env.device)


def push_velocity_range(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice,
  event_name: str,
  stages: list[PushVelocityStage],
) -> dict[str, torch.Tensor]:
  """Increase disturbance strength over training by mutating the push event."""
  del env_ids  # Unused.
  event_cfg = env.event_manager.get_term_cfg(event_name)
  for stage in stages:
    if env.common_step_counter >= stage["step"]:
      event_cfg.params["velocity_range"] = stage["velocity_range"]
  velocity_range = event_cfg.params["velocity_range"]
  return {
    "x_min": torch.tensor(velocity_range["x"][0], device=env.device),
    "x_max": torch.tensor(velocity_range["x"][1], device=env.device),
    "y_min": torch.tensor(velocity_range["y"][0], device=env.device),
    "y_max": torch.tensor(velocity_range["y"][1], device=env.device),
    "yaw_min": torch.tensor(velocity_range["yaw"][0], device=env.device),
    "yaw_max": torch.tensor(velocity_range["yaw"][1], device=env.device),
  }
