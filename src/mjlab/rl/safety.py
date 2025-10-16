from __future__ import annotations

from typing import Callable

import torch


def _wrap_distribution_with_clamp(dist, min_std: float = 1e-6) -> None:
  """Monkey-patch a torch distribution object to clamp std before sampling.

  This is a minimal, safe runtime guard: it does not change training graphs
  except inserting a clamp on the distribution's scale parameter at sampling.
  It supports common attribute names: `scale`, `std`, `stddev`.
  """

  def _make_safe(orig_fn: Callable) -> Callable:
    def _safe(*args, **kwargs):  # type: ignore[no-untyped-def]
      # Try common attribute names used by distributions/diagonal gaussians.
      for name in ("scale", "std", "stddev"):
        if hasattr(dist, name):
          val = getattr(dist, name)
          try:
            setattr(dist, name, torch.clamp(val, min=min_std))
          except Exception:
            pass
      return orig_fn(*args, **kwargs)

    return _safe

  # Patch both sample/rsample if present.
  if hasattr(dist, "sample") and callable(dist.sample):
    dist.sample = _make_safe(dist.sample)  # type: ignore[assignment]
  if hasattr(dist, "rsample") and callable(getattr(dist, "rsample")):
    dist.rsample = _make_safe(dist.rsample)  # type: ignore[assignment]


def clamp_policy_std(policy, min_std: float = 1e-6) -> None:
  """Attach a forward hook to clamp policy distribution std at runtime.

  Works with rsl_rl's ActorCritic by patching `policy.distribution` once and
  re-applying after each forward via a forward hook.
  """

  # Patch current distribution if it already exists.
  dist = getattr(policy, "distribution", None)
  if dist is not None:
    _wrap_distribution_with_clamp(dist, min_std=min_std)

  # Register a forward hook to re-apply after each forward call.
  if hasattr(policy, "register_forward_hook"):
    def _hook(module, inputs, output):  # type: ignore[no-untyped-def]
      d = getattr(module, "distribution", None)
      if d is not None:
        _wrap_distribution_with_clamp(d, min_std=min_std)

    policy.register_forward_hook(_hook)


_PATCHED_FLAG = False


def patch_global_normal_clamp(min_std: float = 1e-6) -> None:
  """Globally monkey-patch torch.distributions.Normal to clamp std on sample.

  This is a last-resort guard to ensure any Normal(...).sample()/rsample() will
  clamp negative scales to a small positive value. Safe to call multiple times.
  """
  global _PATCHED_FLAG
  if _PATCHED_FLAG:
    return
  try:
    from torch.distributions.normal import Normal  # type: ignore

    def _safe_sample(self, sample_shape=torch.Size()):  # type: ignore[no-untyped-def]
      shape = self._extended_shape(sample_shape)
      mean = self.loc.expand(shape)
      std = torch.abs(torch.clamp(self.scale, min=min_std)).expand(shape)
      eps = torch.randn(shape, dtype=self.loc.dtype, device=self.loc.device)
      return mean + std * eps

    def _safe_rsample(self, sample_shape=torch.Size()):  # type: ignore[no-untyped-def]
      shape = self._extended_shape(sample_shape)
      eps = torch.randn(shape, dtype=self.loc.dtype, device=self.loc.device)
      std = torch.abs(torch.clamp(self.scale, min=min_std))
      return self.loc + std * eps

    if hasattr(Normal, "sample"):
      Normal.sample = _safe_sample  # type: ignore[assignment]
    if hasattr(Normal, "rsample"):
      Normal.rsample = _safe_rsample  # type: ignore[assignment]
    _PATCHED_FLAG = True
  except Exception:
    pass


