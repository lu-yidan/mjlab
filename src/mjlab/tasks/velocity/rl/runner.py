import os

import wandb
from rsl_rl.runners import OnPolicyRunner

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.velocity.rl.exporter import (
  attach_onnx_metadata,
  export_velocity_policy_as_onnx,
)
from mjlab.rl.safety import clamp_policy_std, patch_global_normal_clamp


class VelocityOnPolicyRunner(OnPolicyRunner):
  env: RslRlVecEnvWrapper

  def __init__(self, env, train_cfg, log_dir: str | None = None, device: str = "cpu"):
    super().__init__(env, train_cfg, log_dir, device)
    # Clamp policy distribution std to avoid invalid Normal std during sampling.
    try:
      clamp_policy_std(self.alg.policy, min_std=1e-6)
      patch_global_normal_clamp(min_std=1e-6)
    except Exception:
      pass

  def save(self, path: str, infos=None):
    """Save the model and training information."""
    super().save(path, infos)
    if self.logger_type in ["wandb"]:
      policy_path = path.split("model")[0]
      filename = os.path.basename(os.path.dirname(policy_path)) + ".onnx"
      if self.alg.policy.actor_obs_normalization:
        normalizer = self.alg.policy.actor_obs_normalizer
      else:
        normalizer = None
      export_velocity_policy_as_onnx(
        self.alg.policy,
        normalizer=normalizer,
        path=policy_path,
        filename=filename,
      )
      attach_onnx_metadata(
        self.env.unwrapped,
        wandb.run.name,  # type: ignore
        path=policy_path,
        filename=filename,
      )
      wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))
