import gymnasium as gym


gym.register(
  id="Mjlab-Velocity-Flat-AdamLite",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.flat_env_cfg:AdamLiteFlatEnvCfg",
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:AdamLitePPORunnerCfg",
  },
)


gym.register(
  id="Mjlab-Velocity-Flat-AdamLite-Play",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.flat_env_cfg:AdamLiteFlatEnvCfg_PLAY",
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:AdamLitePPORunnerCfg",
  },
)


