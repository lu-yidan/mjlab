"""Unitree G1 recovery environment configurations."""

import math

from mjlab.asset_zoo.robots import G1_ACTION_SCALE, get_g1_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.recovery.recovery_env_cfg import (
  RecoveryFallenPoseCfg,
  make_recovery_env_cfg,
)


KEY_BODY_NAMES = (
  "left_ankle_roll_link",
  "right_ankle_roll_link",
  "left_wrist_yaw_link",
  "right_wrist_yaw_link",
  "left_shoulder_roll_link",
  "right_shoulder_roll_link",
)

G1_FALLEN_POSES = (
  RecoveryFallenPoseCfg(
    name="supine_bent",
    root_pos=(0.0, 0.0, 0.22),
    root_rpy=(math.pi, 0.0, 0.0),
    joint_pos={
      ".*_hip_pitch_joint": -1.05,
      ".*_knee_joint": 2.0,
      ".*_ankle_pitch_joint": -0.95,
      ".*_shoulder_pitch_joint": 0.85,
      "left_shoulder_roll_joint": 0.45,
      "right_shoulder_roll_joint": -0.45,
      ".*_elbow_joint": 1.2,
    },
  ),
  RecoveryFallenPoseCfg(
    name="left_side",
    root_pos=(0.0, 0.0, 0.24),
    root_rpy=(math.pi / 2.0, 0.0, 0.0),
    joint_pos={
      "left_hip_roll_joint": 0.45,
      "right_hip_roll_joint": -0.1,
      "left_hip_pitch_joint": -0.9,
      "right_hip_pitch_joint": -0.5,
      "left_knee_joint": 1.75,
      "right_knee_joint": 1.2,
      "left_ankle_pitch_joint": -0.75,
      "right_ankle_pitch_joint": -0.45,
      "left_shoulder_pitch_joint": 0.6,
      "right_shoulder_pitch_joint": 0.9,
      "left_shoulder_roll_joint": 1.05,
      "right_shoulder_roll_joint": -0.15,
      ".*_elbow_joint": 1.1,
    },
  ),
  RecoveryFallenPoseCfg(
    name="right_side",
    root_pos=(0.0, 0.0, 0.24),
    root_rpy=(-math.pi / 2.0, 0.0, 0.0),
    joint_pos={
      "left_hip_roll_joint": 0.1,
      "right_hip_roll_joint": -0.45,
      "left_hip_pitch_joint": -0.5,
      "right_hip_pitch_joint": -0.9,
      "left_knee_joint": 1.2,
      "right_knee_joint": 1.75,
      "left_ankle_pitch_joint": -0.45,
      "right_ankle_pitch_joint": -0.75,
      "left_shoulder_pitch_joint": 0.9,
      "right_shoulder_pitch_joint": 0.6,
      "left_shoulder_roll_joint": 0.15,
      "right_shoulder_roll_joint": -1.05,
      ".*_elbow_joint": 1.1,
    },
  ),
  RecoveryFallenPoseCfg(
    name="reclined_seated",
    root_pos=(0.0, 0.0, 0.32),
    root_rpy=(2.45, 0.0, 0.0),
    joint_pos={
      ".*_hip_pitch_joint": -1.15,
      ".*_knee_joint": 2.25,
      ".*_ankle_pitch_joint": -1.0,
      ".*_shoulder_pitch_joint": 0.55,
      "left_shoulder_roll_joint": 0.35,
      "right_shoulder_roll_joint": -0.35,
      ".*_elbow_joint": 0.95,
    },
  ),
)


def unitree_g1_flat_recovery_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat-terrain recovery configuration."""
  cfg = make_recovery_env_cfg()

  cfg.scene.entities = {"robot": get_g1_robot_cfg()}

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_ACTION_SCALE

  cfg.observations["actor"].terms["key_body_pos_b"].params["asset_cfg"].body_names = (
    KEY_BODY_NAMES
  )
  cfg.observations["critic"].terms["key_body_pos_b"].params["asset_cfg"].body_names = (
    KEY_BODY_NAMES
  )

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = tuple(
    f"{side}_foot{i}_collision"
    for side in ("left", "right")
    for i in range(1, 8)
  )
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.events["recovery_reset"].params["fallen_pose_library"] = G1_FALLEN_POSES

  cfg.viewer.body_name = "torso_link"

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.scene.num_envs = 48
    cfg.sim.nconmax = 96
    cfg.sim.njmax = 600
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.events.pop("foot_friction", None)
    cfg.events.pop("encoder_bias", None)
    cfg.events.pop("base_com", None)
    cfg.curriculum = {}
    cfg.teacher.motion_path = ""
    cfg.teacher.reference_probability = 0.0
    cfg.events["recovery_reset"].params["fallen_pose_probability"] = 1.0
    cfg.events["recovery_reset"].params["random_fall_probability"] = 0.0
    cfg.events["recovery_reset"].params["motion_fallen_zero_velocity"] = True
    cfg.events["recovery_reset"].params["fallen_pose_height_offset"] = 0.04
    cfg.events["recovery_reset"].params["fallen_pose_xy_jitter"] = 0.0
    cfg.events["recovery_reset"].params["fallen_pose_yaw_jitter"] = 0.0
    cfg.events["recovery_reset"].params["fallen_pose_joint_jitter"] = 0.0

  return cfg
