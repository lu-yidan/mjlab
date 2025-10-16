"""PND Adam Lite robot config and helpers.

This module minimally integrates the `adam_lite.xml` MJCF into mjlab by
providing a `get_spec()` function and an `EntityCfg` describing the robot.

You can extend this later with explicit actuator models and per-joint action
scales (see `unitree_g1/g1_constants.py` for a comprehensive example).
"""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import ActuatorCfg


# Path to the MJCF file bundled with this repository.
ADAM_LITE_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "pnd_adam_lite" / "adam_lite.xml"
)
assert ADAM_LITE_XML.exists(), f"adam_lite.xml not found at: {ADAM_LITE_XML}"


def get_assets(meshdir: str) -> dict[str, bytes]:
  """Collect external assets referenced by the MJCF into MuJoCo's asset dict.

  This copies meshes/textures from the robot's assets directory into the
  runtime asset dictionary so `mujoco.MjSpec` can be compiled anywhere.
  """
  assets: dict[str, bytes] = {}
  # The MJCF uses compiler meshdir="assets", so we must load the files from
  # the local "assets" subdirectory and prefix keys with meshdir.
  update_assets(assets, ADAM_LITE_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  """Load the adam_lite MJCF and attach its assets."""
  spec = mujoco.MjSpec.from_file(str(ADAM_LITE_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


# Minimal articulation: we rely on actuators defined in the MJCF for now.
ADAM_LITE_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    # Lower body
    ActuatorCfg(joint_names_expr=[r"^hipPitch_.*$"],  effort_limit=230.0, stiffness=305.0, damping=6.1),
    ActuatorCfg(joint_names_expr=[r"^hipRoll_.*$"],   effort_limit=160.0, stiffness=700.0, damping=30.0),
    ActuatorCfg(joint_names_expr=[r"^hipYaw_.*$"],    effort_limit=105.0, stiffness=405.0, damping=6.1),
    ActuatorCfg(joint_names_expr=[r"^kneePitch_.*$"], effort_limit=230.0, stiffness=305.0, damping=6.1),
    ActuatorCfg(joint_names_expr=[r"^anklePitch_.*$"],effort_limit=40.0,  stiffness=20.0,  damping=2.5),
    ActuatorCfg(joint_names_expr=[r"^ankleRoll_.*$"], effort_limit=12.0,  stiffness=0.0,   damping=0.35),
    # Waist
    ActuatorCfg(joint_names_expr=[r"^waistRoll$"],    effort_limit=110.0, stiffness=405.0, damping=6.1),
    ActuatorCfg(joint_names_expr=[r"^waistPitch$"],   effort_limit=110.0, stiffness=405.0, damping=6.1),
    ActuatorCfg(joint_names_expr=[r"^waistYaw$"],     effort_limit=110.0, stiffness=205.0, damping=4.1),
    # Arms (Left/Right)
    ActuatorCfg(joint_names_expr=[r"^shoulderPitch_.*$"], effort_limit=65.0, stiffness=18.0, damping=0.9),
    ActuatorCfg(joint_names_expr=[r"^shoulderRoll_.*$"],  effort_limit=65.0, stiffness=9.0,  damping=0.9),
    ActuatorCfg(joint_names_expr=[r"^shoulderYaw_.*$"],   effort_limit=65.0, stiffness=9.0,  damping=0.9),
    ActuatorCfg(joint_names_expr=[r"^elbow_.*$"],         effort_limit=30.0, stiffness=9.0,  damping=0.9),
  ),
  soft_joint_pos_limit_factor=0.9,
)


# Entity configuration consumed by scenes/managers.
ADAM_LITE_ROBOT_CFG = EntityCfg(
  # Default joint pose from your adam_config.py
  init_state=EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.89),
    joint_pos={
      # Left leg
      "^hipPitch_Left$": -0.32,
      "^hipRoll_Left$": 0.0,
      "^hipYaw_Left$": -0.18,
      "^kneePitch_Left$": 0.66,
      "^anklePitch_Left$": -0.39,
      "^ankleRoll_Left$": 0.0,
      # Right leg
      "^hipPitch_Right$": -0.32,
      "^hipRoll_Right$": 0.0,
      "^hipYaw_Right$": 0.18,
      "^kneePitch_Right$": 0.66,
      "^anklePitch_Right$": -0.39,
      "^ankleRoll_Right$": 0.0,
      # Waist
      "^waistRoll$": 0.0,
      "^waistPitch$": 0.0,
      "^waistYaw$": 0.0,
      # Left arm
      "^shoulderPitch_Left$": 0.0,
      "^shoulderRoll_Left$": 0.1,
      "^shoulderYaw_Left$": 0.0,
      "^elbow_Left$": -0.3,
      # Right arm
      "^shoulderPitch_Right$": 0.0,
      "^shoulderRoll_Right$": -0.1,
      "^shoulderYaw_Right$": 0.0,
      "^elbow_Right$": -0.3,
    },
    joint_vel={".*": 0.0},
  ),
  collisions=(),
  spec_fn=get_spec,
  articulation=ADAM_LITE_ARTICULATION,
)


# Optional per-joint action scaling; keep empty for now.
ADAM_LITE_ACTION_SCALE: dict[str, float] = {}


