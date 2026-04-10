# G1 Recovery Migration Notes

This document records the changes made to support a deployment-time
reference-free `G1 Recovery` task in `mjlab`, along with the current
verification status and the commands used to inspect converted teacher motions.

## Summary

The goal of this migration was not to directly port the Isaac Lab AMP stack from
`amp_rec`, but to rebuild the recovery task semantics inside `mjlab`'s native
MuJoCo-Warp manager framework.

The resulting design has these properties:

- `mjlab` now has a standalone recovery task: `Mjlab-Recovery-Flat-Unitree-G1`
- the actor does not consume reference motion observations
- play/deployment does not require a motion artifact
- converted get-up motions are used only as optional teacher states for reset
  initialization during training

## New Task Code

The following task code was added under `src/mjlab/tasks/recovery/`:

- `recovery_env_cfg.py`
  - defines the base recovery task configuration
  - introduces `RecoveryTeacherCfg` for optional teacher-motion resets
- `mdp/observations.py`
  - root orientation without yaw
  - key body positions in the root frame
  - root height observation
- `mdp/rewards.py`
  - rise/stand shaping
  - orientation and height targets
  - stabilization rewards for base linear/angular velocity
- `mdp/terminations.py`
  - safety termination for excessive joint velocity
- `mdp/curriculums.py`
  - teacher reset probability schedule
  - push disturbance schedule
- `mdp/events.py`
  - `RecoveryReset`, which samples random fallen states
  - optional motion-teacher reset path based on converted `.npz` clips
- `config/g1/env_cfgs.py`
  - G1-specific scene, sensors, body names, and play-mode overrides
- `config/g1/rl_cfg.py`
  - PPO runner configuration for recovery
- `config/g1/__init__.py`
  - registers `Mjlab-Recovery-Flat-Unitree-G1`

## Motion Conversion Bridge

Two scripts were added under `src/mjlab/scripts/`:

- `amp_pkl_to_npz.py`
  - converts `amp_rec` `.pkl` get-up motions into a compact `.npz` format
  - stores only the root and joint state needed by recovery teacher resets
  - reorders `dof_pos` from `amp_rec`'s `lab_dof_names` order into `mjlab`'s
    native G1 joint order before saving
- `play_recovery_motion.py`
  - plays converted `.npz` clips directly on the G1 robot in `mjlab`
  - useful for checking whether the converted motion looks physically sensible

The converted motion directory currently used is:

`artifacts/recovery_motions/g1_amp_get_up`

## Validation Performed

The following checks were completed:

- recovery task config tests passed
- generic task config tests still passed
- the new recovery environment could be instantiated successfully
- the new recovery environment could also be instantiated with
  `teacher.motion_path` pointing at the converted motion directory
- the conversion script CLI was exercised successfully
- all `.pkl` get-up clips under the `amp_rec` get-up directory were converted to
  `.npz`

## Pitfalls And Fixes

These were real issues hit during migration and playback. They are documented
here so future contributors or agents do not repeat the same mistakes.

### 1. Motion looked badly twisted in `mjlab`

Symptom:

- the robot posture was severely distorted during playback
- the body did not just look slightly off; joints appeared globally scrambled

Root cause:

- the source `amp_rec` `.pkl` files do not store `dof_pos` in the same order as
  `mjlab`'s native G1 joint order
- the `amp_rec` retarget pipeline exports `dof_pos` using the `lab_dof_names`
  order from `scripts/tools/retarget/config/g1_29dof.yaml`
- `mjlab` G1 playback expects the joint order used by the MuJoCo robot model

Important detail:

- this issue was initially easy to confuse with quaternion convention bugs
- in this case the main problem was not `wxyz` vs `xyzw`
- the dominant bug was joint-order mismatch

Fix applied:

- `src/mjlab/scripts/amp_pkl_to_npz.py` now explicitly reorders `dof_pos`
  from `amp_rec`'s `lab_dof_names` order into `mjlab`'s G1 joint order
- the converted `.npz` files were regenerated after the fix
- the converter now also writes `joint_names` into the output archive for
  easier inspection

Practical rule:

- if playback looks globally twisted, check joint order before investigating
  quaternion math

### 2. Windowed playback failed on camera setup

Symptom:

- `play_recovery_motion.py` failed before playback started
- error looked like `KeyError: Invalid name 'torso_link'`

Root cause:

- the MuJoCo scene body names are namespaced, e.g. `robot/torso_link`
- the first viewer implementation tried to resolve `torso_link` directly from
  `mjModel`

Fix applied:

- `src/mjlab/scripts/play_recovery_motion.py` now resolves the tracked camera
  body through the `robot` entity instead of hardcoding a raw model body name

Practical rule:

- when using scene entities in `mjlab`, prefer resolving body IDs through the
  entity API rather than assuming raw `mjModel` names

### 3. Headless playback is a good first check

Observation:

- headless playback succeeded before the GUI path was fully correct
- this made it easy to separate simulation/data issues from viewer-only issues

Practical rule:

- when validating converted motions, first run:

```bash
uv run python -m mjlab.scripts.play_recovery_motion \
  --motion-path "/home/ydlu/workspace/mjlab/artifacts/recovery_motions/g1_amp_get_up" \
  --clip "fallAndGetUp2_subject2_1200_1370" \
  --headless \
  --max-frames 120
```

- if headless works but GUI fails, suspect the viewer path first

## How To Play Converted Motions

### 1. Play the first converted clip in a directory

```bash
cd /home/ydlu/workspace/mjlab

uv run python -m mjlab.scripts.play_recovery_motion \
  --motion-path "/home/ydlu/workspace/mjlab/artifacts/recovery_motions/g1_amp_get_up"
```

If `--clip` is not specified, the script plays the first `.npz` clip in sorted
order.

### 2. Play a specific clip

```bash
cd /home/ydlu/workspace/mjlab

uv run python -m mjlab.scripts.play_recovery_motion \
  --motion-path "/home/ydlu/workspace/mjlab/artifacts/recovery_motions/g1_amp_get_up" \
  --clip "fallAndGetUp2_subject2_1200_1370"
```

### 3. Run a headless smoke test

```bash
cd /home/ydlu/workspace/mjlab

uv run python -m mjlab.scripts.play_recovery_motion \
  --motion-path "/home/ydlu/workspace/mjlab/artifacts/recovery_motions/g1_amp_get_up" \
  --clip "fallAndGetUp2_subject2_1200_1370" \
  --headless \
  --max-frames 120
```

### 4. Play a single `.npz` file directly

```bash
cd /home/ydlu/workspace/mjlab

uv run python -m mjlab.scripts.play_recovery_motion \
  --motion-path "/home/ydlu/workspace/mjlab/artifacts/recovery_motions/g1_amp_get_up/fallAndGetUp2_subject2_1200_1370.npz"
```

## What To Look For During Playback

When checking whether the converted trajectory is reasonable, focus on:

- root orientation continuity
- whether the robot starts from a plausible fallen pose
- whether joint motion is smooth rather than jittering
- whether the get-up transition qualitatively matches the source motion
- whether there are obvious left/right flips or quaternion errors

If the motion looks wrong, the first suspects are:

- joint order mismatch
- quaternion convention mismatch
- velocity reconstruction mismatch
- root height offset mismatch

## Training With Teacher Motion Reset

Once playback looks correct, a baseline training command is:

```bash
cd /home/ydlu/workspace/mjlab

uv run train Mjlab-Recovery-Flat-Unitree-G1 \
  --env.teacher.motion-path "/home/ydlu/workspace/mjlab/artifacts/recovery_motions/g1_amp_get_up" \
  --env.scene.num-envs 4096
```

This uses teacher motion only for reset-state sampling. The trained actor still
does not receive reference motion as input.

## Current Limitations

- AMP discriminator logic has not been ported into `mjlab`
- the current converted motion format is aimed at recovery reset/playback, not
  the full `mjlab` tracking pipeline
- the playback tool is intended for quick qualitative inspection, not for exact
  tracking-metric evaluation
