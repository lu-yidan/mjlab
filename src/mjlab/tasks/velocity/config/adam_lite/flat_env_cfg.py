from dataclasses import dataclass, replace

from mjlab.asset_zoo.robots.pnd_adam_lite.adam_lite_constants import (
  ADAM_LITE_ACTION_SCALE,
  ADAM_LITE_ROBOT_CFG,
)
from mjlab.tasks.velocity.velocity_env_cfg import (
  LocomotionVelocityEnvCfg,
)


@dataclass
class AdamLiteFlatEnvCfg(LocomotionVelocityEnvCfg):
  def __post_init__(self):
    super().__post_init__()

    # Insert robot into the scene.
    robot_cfg = replace(ADAM_LITE_ROBOT_CFG)
    self.scene.entities = {"robot": robot_cfg}

    # Optional: per-joint scaling.
    if ADAM_LITE_ACTION_SCALE:
      self.actions.joint_pos.scale = ADAM_LITE_ACTION_SCALE
    else:
      # 更小的默认动作尺度，先稳住训练
      self.actions.joint_pos.scale = 0.2

    # Camera跟随主体。你提到base像是"pelvis"，这里设置为它。
    self.viewer.body_name = "pelvis"

    # 姿态奖励std（关节名正则 -> 标准差）。必须是字典，不能是空列表。
    # 取自 G1 的经验并结合 adam_config.py 的关节分组，先给稳健的初值。
    self.rewards.pose.params["std"] = {
      # Lower body
      r".*hipPitch.*": 0.3,
      r".*hipRoll.*": 0.15,
      r".*hipYaw.*": 0.15,
      r".*kneePitch.*": 0.35,
      r".*anklePitch.*": 0.25,
      r".*ankleRoll.*": 0.10,
      # Waist
      r".*waistYaw.*": 0.15,
      r".*waistRoll.*": 0.08,
      r".*waistPitch.*": 0.10,
      # Arms
      r".*shoulderPitch.*": 0.35,
      r".*shoulderRoll.*": 0.15,
      r".*shoulderYaw.*": 0.10,
      r".*elbow.*": 0.25,
    }


@dataclass
class AdamLiteFlatEnvCfg_PLAY(AdamLiteFlatEnvCfg):
  def __post_init__(self):
    super().__post_init__()
    # 平地可视化：与 G1 的 PLAY 一致
    assert self.scene.terrain is not None
    self.scene.terrain.terrain_type = "plane"
    self.scene.terrain.terrain_generator = None
    self.curriculum.terrain_levels = None
    self.curriculum.command_vel = None
    assert self.events.push_robot is not None
    self.events.push_robot.params["velocity_range"] = {"x": (-0.2, 0.2), "y": (-0.2, 0.2)}
    self.episode_length_s = int(1e9)

    # 姿态奖励的std需要是“关节名正则 -> 标准差”的字典。
    # 参考 G1 的设置，并结合 adam_config.py 的关节分组给出稳健的初始值。
    self.rewards.pose.params["std"] = {
      # Lower body
      r".*hipPitch.*": 0.3,
      r".*hipRoll.*": 0.15,
      r".*hipYaw.*": 0.15,
      r".*kneePitch.*": 0.35,
      r".*anklePitch.*": 0.25,
      r".*ankleRoll.*": 0.10,
      # Waist
      r".*waistYaw.*": 0.15,
      r".*waistRoll.*": 0.12,
      r".*waistPitch.*": 0.15,
      # Arms
      r".*shoulderPitch.*": 0.35,
      r".*shoulderRoll.*": 0.15,
      r".*shoulderYaw.*": 0.10,
      r".*elbow.*": 0.25,
      r".*wrist.*": 0.30,
    }


