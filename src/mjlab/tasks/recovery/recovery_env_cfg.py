"""Recovery task configuration.

This task trains a policy to stand up from fallen states without consuming
reference motion observations at deployment time. Optional teacher motions can
be used only for reset initialization during training.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.recovery import mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig


@dataclass(kw_only=True)
class RecoveryTeacherCfg:
  """Optional motion teacher used only for recovery-state initialization."""

  motion_path: str = ""
  motion_weights: dict[str, float] = field(default_factory=dict)
  reference_probability: float = 0.7
  min_progress: float = 0.08
  max_progress: float = 0.6
  height_offset: float = 0.05


@dataclass(kw_only=True)
class RecoveryFallenPoseCfg:
  """A physically plausible fallen pose used for reset or evaluation."""

  name: str
  root_pos: tuple[float, float, float]
  root_rpy: tuple[float, float, float]
  joint_pos: dict[str, float]
  joint_vel: dict[str, float] = field(default_factory=lambda: {".*": 0.0})


@dataclass(kw_only=True)
class RecoveryEnvCfg(ManagerBasedRlEnvCfg):
  """Extension of the base RL env config with recovery-specific teacher options."""

  teacher: RecoveryTeacherCfg = field(default_factory=RecoveryTeacherCfg)


def make_recovery_env_cfg() -> RecoveryEnvCfg:
  """Create the base recovery environment configuration."""

  actor_terms = {
    "base_ang_vel": ObservationTermCfg(
      func=envs_mdp.base_ang_vel,
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "root_local_rot_tan_norm": ObservationTermCfg(
      func=mdp.root_local_rot_tan_norm,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "joint_pos": ObservationTermCfg(
      func=envs_mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=envs_mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
    ),
    "key_body_pos_b": ObservationTermCfg(
      func=mdp.key_body_pos_b,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=())},
      noise=Unoise(n_min=-0.08, n_max=0.08),
    ),
    "root_height": ObservationTermCfg(
      func=mdp.root_height,
      noise=Unoise(n_min=-0.02, n_max=0.02),
    ),
    "actions": ObservationTermCfg(func=envs_mdp.last_action),
  }

  critic_terms = {
    "base_lin_vel": ObservationTermCfg(func=envs_mdp.base_lin_vel),
    **actor_terms,
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
      history_length=4,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
      history_length=4,
    ),
  }

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": mdp.RecoveryJointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.5,
      use_default_offset=True,
      unactuated_steps=30,
    )
  }

  events = {
    "recovery_reset": EventTermCfg(
      func=mdp.RecoveryReset,
      mode="reset",
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "base_height_range": (0.18, 0.38),
        "joint_position_range": (-0.45, 0.45),
        "joint_velocity_range": (-1.0, 1.0),
        "fallen_pose_library": (),
        "fallen_pose_probability": 0.5,
        "random_fall_probability": 0.0,
        "motion_fallen_progress_range": (0.0, 0.08),
        "motion_fallen_zero_velocity": True,
        "fallen_pose_height_offset": 0.02,
        "fallen_pose_xy_jitter": 0.01,
        "fallen_pose_yaw_jitter": 0.1,
        "fallen_pose_joint_jitter": 0.03,
        "teacher_post_reset_settle_steps": 0,
        "fallen_post_reset_settle_steps": 8,
        "random_post_reset_settle_steps": 8,
        "root_velocity_range": {
          "x": (-0.2, 0.2),
          "y": (-0.2, 0.2),
          "z": (-0.2, 0.2),
          "roll": (-0.5, 0.5),
          "pitch": (-0.5, 0.5),
          "yaw": (-0.5, 0.5),
        },
      },
    ),
    "upward_assist": EventTermCfg(
      func=mdp.UpwardAssistance,
      mode="step",
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
        "force": 200.0,
        "assist_start_steps": 30,
        "assist_until_height": 0.7,
        "orientation_z_threshold": -0.7,
        "use_orientation_gate": False,
      },
    ),
    "push_robot": EventTermCfg(
      func=envs_mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(5.0, 8.0),
      params={
        "velocity_range": {
          "x": (-0.15, 0.15),
          "y": (-0.15, 0.15),
          "yaw": (-0.35, 0.35),
        }
      },
    ),
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),
        "operation": "abs",
        "ranges": (0.3, 1.2),
        "shared_random": True,
      },
    ),
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=dr.encoder_bias,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "bias_range": (-0.01, 0.01),
      },
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),
        "operation": "add",
        "ranges": {
          0: (-0.025, 0.025),
          1: (-0.05, 0.05),
          2: (-0.05, 0.05),
        },
      },
    ),
  }

  rewards = {
    "alive": RewardTermCfg(func=envs_mdp.is_alive, weight=0.25),
    "base_height_progress": RewardTermCfg(
      func=mdp.base_height_progress,
      weight=3.0,
      params={
        "minimum_height": 0.18,
        "target_height": 0.75,
      },
    ),
    "target_orientation": RewardTermCfg(
      func=mdp.target_orientation,
      weight=2.0,
      params={
        "target_base_height_phase3": 0.65,
        "std": 0.35,
      },
    ),
    "target_base_height": RewardTermCfg(
      func=mdp.target_base_height,
      weight=5.0,
      params={
        "base_height_target": 0.75,
        "target_base_height_phase3": 0.65,
        "std": 0.05,
      },
    ),
    "lin_vel_xy": RewardTermCfg(
      func=mdp.lin_vel_xy,
      weight=1.5,
      params={
        "target_base_height_phase3": 0.65,
        "std": 0.35,
      },
    ),
    "ang_vel_xy": RewardTermCfg(
      func=mdp.ang_vel_xy,
      weight=1.5,
      params={
        "target_base_height_phase3": 0.65,
        "std": 0.45,
      },
    ),
    "stand_bonus": RewardTermCfg(
      func=mdp.stand_bonus,
      weight=2.0,
      params={
        "stand_height": 0.72,
        "upright_threshold": 0.9,
      },
    ),
    "late_phase_orientation": RewardTermCfg(
      func=mdp.late_phase_orientation,
      weight=1.0,
      params={
        "activation_height": 0.68,
        "full_weight_height": 0.75,
        "std": 0.2,
      },
    ),
    "late_phase_base_motion": RewardTermCfg(
      func=mdp.late_phase_base_motion,
      weight=1.0,
      params={
        "activation_height": 0.68,
        "full_weight_height": 0.75,
        "lin_vel_std": 0.15,
        "ang_vel_std": 0.2,
      },
    ),
    "late_phase_ankle_vel_l2": RewardTermCfg(
      func=mdp.late_phase_joint_vel_l2,
      weight=-2.0e-3,
      params={
        "activation_height": 0.66,
        "full_weight_height": 0.74,
        "asset_cfg": SceneEntityCfg(
          "robot",
          joint_names=(
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
          ),
          preserve_order=True,
        ),
      },
    ),
    "late_phase_ankle_home": RewardTermCfg(
      func=mdp.late_phase_posture,
      weight=0.45,
      params={
        "activation_height": 0.68,
        "full_weight_height": 0.76,
        "std": 0.2,
        "asset_cfg": SceneEntityCfg(
          "robot",
          joint_names=(
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
          ),
          preserve_order=True,
        ),
      },
    ),
    "late_phase_hip_home": RewardTermCfg(
      func=mdp.late_phase_posture,
      weight=0.35,
      params={
        "activation_height": 0.69,
        "full_weight_height": 0.76,
        "std": 0.25,
        "asset_cfg": SceneEntityCfg(
          "robot",
          joint_names=(
            "left_hip_yaw_joint",
            "left_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_hip_roll_joint",
          ),
          preserve_order=True,
        ),
      },
    ),
    "late_phase_arm_home": RewardTermCfg(
      func=mdp.late_phase_posture,
      weight=0.35,
      params={
        "activation_height": 0.7,
        "full_weight_height": 0.76,
        "std": 0.35,
        "asset_cfg": SceneEntityCfg(
          "robot",
          joint_names=(
            ".*_shoulder_.*_joint",
            ".*_elbow_joint",
            ".*_wrist_.*_joint",
          ),
          preserve_order=True,
        ),
      },
    ),
    "self_collisions": RewardTermCfg(
      func=mdp.self_collision_cost,
      weight=-1.0,
      params={"sensor_name": "self_collision", "force_threshold": 10.0},
    ),
    "joint_acc_l2": RewardTermCfg(func=envs_mdp.joint_acc_l2, weight=-2.5e-7),
    "action_rate_l2": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.01),
    "action_acc_l2": RewardTermCfg(func=envs_mdp.action_acc_l2, weight=-2.5e-3),
    "joint_torques_l2": RewardTermCfg(func=envs_mdp.joint_torques_l2, weight=-2.5e-6),
    "joint_pos_limits": RewardTermCfg(
      func=envs_mdp.joint_pos_limits,
      weight=-100.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    "root_height_floor": TerminationTermCfg(
      func=mdp.root_height_below_minimum_after_grace,
      params={"minimum_height": 0.08, "min_steps_before_check": 40},
    ),
    "nan_detection": TerminationTermCfg(func=envs_mdp.nan_detection),
    "joint_vel_limit": TerminationTermCfg(
      func=mdp.joint_vel_out_of_manual_limit,
      params={"max_velocity": 40.0},
    ),
  }

  curriculum = {
    "reference_probability": CurriculumTermCfg(
      func=mdp.reference_reset_probability,
      params={
        "event_name": "recovery_reset",
        "stages": [
          {"step": 0, "probability": 0.7},
          {"step": 5_000 * 24, "probability": 0.5},
          {"step": 15_000 * 24, "probability": 0.15},
        ],
      },
    ),
    "upward_assistance_force": CurriculumTermCfg(
      func=mdp.upward_assistance_force,
      params={
        "event_name": "upward_assist",
        "success_body_name": "torso_link",
        "success_height": 0.78,
        "success_upright_threshold": 0.8,
        "decrement": 10.0,
        "min_force": 0.0,
      },
    ),
    "push_velocity": CurriculumTermCfg(
      func=mdp.push_velocity_range,
      params={
        "event_name": "push_robot",
        "stages": [
          {
            "step": 0,
            "velocity_range": {
              "x": (-0.05, 0.05),
              "y": (-0.05, 0.05),
              "yaw": (-0.15, 0.15),
            },
          },
          {
            "step": 8_000 * 24,
            "velocity_range": {
              "x": (-0.1, 0.1),
              "y": (-0.1, 0.1),
              "yaw": (-0.25, 0.25),
            },
          },
          {
            "step": 16_000 * 24,
            "velocity_range": {
              "x": (-0.15, 0.15),
              "y": (-0.15, 0.15),
              "yaw": (-0.35, 0.35),
            },
          },
        ],
      },
    ),
  }

  return RecoveryEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      num_envs=1,
      env_spacing=2.5,
    ),
    observations=observations,
    actions=actions,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum=curriculum,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",
      distance=2.4,
      elevation=-5.0,
      azimuth=120.0,
    ),
    sim=SimulationCfg(
      nconmax=45,
      njmax=350,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
      ),
    ),
    decimation=4,
    episode_length_s=12.0,
  )
