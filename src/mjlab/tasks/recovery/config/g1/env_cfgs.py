"""Unitree G1 recovery environment configurations."""

from mjlab.asset_zoo.robots import G1_ACTION_SCALE, get_g1_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.recovery.recovery_env_cfg import make_recovery_env_cfg


KEY_BODY_NAMES = (
  "left_ankle_roll_link",
  "right_ankle_roll_link",
  "left_wrist_yaw_link",
  "right_wrist_yaw_link",
  "left_shoulder_roll_link",
  "right_shoulder_roll_link",
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

  cfg.viewer.body_name = "torso_link"

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.scene.num_envs = 48
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.teacher.motion_path = ""
    cfg.teacher.reference_probability = 0.0

  return cfg
