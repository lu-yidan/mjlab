from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import tyro

import mjlab

EXPECTED_COLUMNS = 36


@dataclass(frozen=True)
class MotionCsvStats:
  path: str
  frames: int
  columns: int
  duration_s: float
  compatible_with_mjlab: bool
  root_z_min: float | None
  root_z_max: float | None
  quat_norm_mean: float | None
  quat_norm_max_abs_error: float | None
  joint_abs_max: float | None
  error: str | None = None


def _load_csv(path: Path) -> np.ndarray:
  data = np.loadtxt(path, delimiter=",", dtype=np.float32)
  return np.atleast_2d(data)


def analyze_csv(path: Path, input_fps: float) -> MotionCsvStats:
  try:
    data = _load_csv(path)
  except Exception as exc:  # pragma: no cover - trivial error plumbing
    return MotionCsvStats(
      path=str(path),
      frames=0,
      columns=0,
      duration_s=0.0,
      compatible_with_mjlab=False,
      root_z_min=None,
      root_z_max=None,
      quat_norm_mean=None,
      quat_norm_max_abs_error=None,
      joint_abs_max=None,
      error=str(exc),
    )

  frames, columns = data.shape
  compatible = columns == EXPECTED_COLUMNS

  root_z_min = float(data[:, 2].min()) if columns >= 3 else None
  root_z_max = float(data[:, 2].max()) if columns >= 3 else None
  quat_norm_mean = None
  quat_norm_max_abs_error = None
  if columns >= 7:
    quat_norms = np.linalg.norm(data[:, 3:7], axis=1)
    quat_norm_mean = float(quat_norms.mean())
    quat_norm_max_abs_error = float(np.abs(quat_norms - 1.0).max())

  joint_abs_max = float(np.abs(data[:, 7:]).max()) if columns > 7 else None

  return MotionCsvStats(
    path=str(path),
    frames=frames,
    columns=columns,
    duration_s=frames / input_fps,
    compatible_with_mjlab=compatible,
    root_z_min=root_z_min,
    root_z_max=root_z_max,
    quat_norm_mean=quat_norm_mean,
    quat_norm_max_abs_error=quat_norm_max_abs_error,
    joint_abs_max=joint_abs_max,
    error=None,
  )


def _discover_csv_files(input_path: Path, glob_pattern: str) -> list[Path]:
  if input_path.is_file():
    return [input_path]
  if input_path.is_dir():
    return sorted(input_path.rglob(glob_pattern))
  raise FileNotFoundError(f"Input path does not exist: {input_path}")


def _format_optional(value: float | None) -> str:
  if value is None:
    return "n/a"
  return f"{value:.4f}"


def main(
  input_path: str,
  input_fps: float = 100.0,
  glob_pattern: str = "*.csv",
  limit: int | None = 20,
  output_json: str | None = None,
):
  """Analyze tracking CSV files before conversion.

  Args:
    input_path: CSV file or directory containing CSV files.
    input_fps: Assumed source frame rate used for duration estimates.
    glob_pattern: Glob used when `input_path` is a directory.
    limit: Maximum number of file summaries to print. Use `None` for all files.
    output_json: Optional path to save the full report as JSON.
  """
  resolved_input = Path(input_path).expanduser()
  csv_files = _discover_csv_files(resolved_input, glob_pattern)
  if not csv_files:
    raise FileNotFoundError(
      f"No CSV files found under {resolved_input} with pattern '{glob_pattern}'."
    )

  stats = [analyze_csv(path, input_fps=input_fps) for path in csv_files]
  compatible_count = sum(stat.compatible_with_mjlab for stat in stats)
  error_count = sum(stat.error is not None for stat in stats)
  total_frames = sum(stat.frames for stat in stats)
  total_duration_s = sum(stat.duration_s for stat in stats)

  print(f"Analyzed {len(stats)} CSV file(s)")
  print(f"Assumed input FPS: {input_fps:g}")
  print(f"Compatible with mjlab CSV layout: {compatible_count}/{len(stats)}")
  print(f"Files with read errors: {error_count}")
  print(f"Total frames: {total_frames}")
  print(f"Total duration: {total_duration_s:.2f}s")

  ranked_stats = sorted(stats, key=lambda stat: stat.frames, reverse=True)
  shown_stats = ranked_stats if limit is None else ranked_stats[:limit]

  print("")
  print("Per-file summary:")
  for stat in shown_stats:
    print(
      "  "
      f"{stat.path}: frames={stat.frames}, cols={stat.columns}, "
      f"duration={stat.duration_s:.2f}s, compatible={stat.compatible_with_mjlab}, "
      f"root_z=[{_format_optional(stat.root_z_min)}, {_format_optional(stat.root_z_max)}], "
      f"quat_norm_mean={_format_optional(stat.quat_norm_mean)}, "
      f"quat_norm_max_err={_format_optional(stat.quat_norm_max_abs_error)}, "
      f"joint_abs_max={_format_optional(stat.joint_abs_max)}"
      + (f", error={stat.error}" if stat.error is not None else "")
    )

  if output_json is not None:
    output_path = Path(output_json).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
      json.dumps([asdict(stat) for stat in stats], indent=2), encoding="utf-8"
    )
    print("")
    print(f"Saved JSON report to {output_path}")


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
