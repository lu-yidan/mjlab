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

## Current Algorithm Status

It is important to be precise about what has and has not been migrated.

The current `mjlab` recovery task is:

- a native `mjlab` recovery PPO task
- reference-free at inference time
- optionally initialized from teacher motion states during reset

The current `mjlab` recovery task is **not yet**:

- an AMP task
- a discriminator-based style-learning pipeline
- a direct port of `AMPRunner` / `PPOAMP` from `amp_rec`

In other words, this migration currently preserves the **task semantics** of
recovery/get-up, but not the original AMP algorithm stack.

This distinction matters when interpreting training curves:

- `amp_rec` TensorBoard logs are still useful as a stability and monitoring
  reference
- but `mjlab` reward and loss magnitudes should not be expected to numerically
  match the original AMP run

## What Teacher Reset Means

`teacher reset` does **not** mean feeding a reference trajectory into the actor
during rollout.

Instead, it means:

- at episode reset, some environments are initialized from a sampled state on a
  converted get-up motion clip
- the rest of the environments are initialized from random fallen states
- after reset, the policy acts only on proprioceptive/body-state observations

Why this is useful:

- it makes early recovery learning easier by exposing the policy to successful
  near-recovery states
- it still preserves a deployment-time reference-free policy
- it provides a curriculum path from "continue a plausible get-up" to
  "recover from arbitrary falls"

The intended use is:

- use teacher motion during training only
- decay its use over time
- do not depend on it for deployment

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

## Reset Strategy

The recovery task should not rely on a pure "random air-drop" reset. That
approach is too easy to make physically inconsistent and can produce:

- partial ground penetration at reset
- explosive contacts on the first physics step
- misleading checkpoint visualization where the robot appears to fail before the
  policy has a meaningful chance to act

The current recovery reset strategy is a mixture of:

- teacher motion reset states
- stable fallen poses for evaluation/fallback
- only a tiny or zero random-fall component by default

### Training reset

When `teacher.motion_path` is provided:

- `teacher` states are sampled from the converted get-up motions using the main
  progress window in `RecoveryTeacherCfg`
- `fallen` states are sampled from the same motion library, but from an early
  progress window near the lying/fallen portion of the clip
- those fallen states are zero-velocity by default to avoid immediately
  exploding contacts

This keeps training physically grounded while still exposing the policy to
realistic recovery starts.

### Play / evaluation reset

For `play=True`:

- startup domain randomization is disabled
- push events are disabled
- teacher resets are disabled by default
- reset uses a small library of canonical, physically plausible fallen poses

This is intentionally different from training because it makes checkpoint
inspection easier and less noisy.

If you want `play` to use motion-derived fallen states instead of the fallback
canonical poses, pass:

```bash
--teacher-motion-path "/path/to/converted/recovery_motions"
```

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
- canonical fallback fallen poses were previewed; some handwritten poses still
  showed penetration artifacts and therefore should not be treated as the main
  reset source
- motion-derived fallen-state previews looked substantially more physical in
  practice and are the preferred source for reset inspection
- settle-step visualization support was added to the fallen-pose preview script
  so reset states can be inspected over the first few physics steps instead of
  only as a static pose

## Experiment Record

The following recovery runs are the main reference points so far.

### 1. Old reset + aggressive PPO

Run:

- `/home/ydlu/workspace/mjlab/logs/rsl_rl/g1_recovery/2026-04-10_21-29-51`

Observed behavior:

- many environments appeared to "bounce" or catch an unstable reset
- policies could sometimes stabilize near-standing states
- once genuinely dragged down, recovery was poor
- higher checkpoints often looked more jittery than mid-run checkpoints

End-of-run summary:

- `Train/mean_reward`: `23.24`
- `Train/mean_episode_length`: `144.84`
- `Policy/mean_std`: `2.10`
- `Episode_Reward/stand_bonus`: `0.46`
- `Episode_Termination/joint_vel_limit`: `21.13`

Interpretation:

- reset improvements were still incomplete
- PPO became too aggressive and learned high-variance solutions

### 2. New reset + still-aggressive PPO

Run:

- `/home/ydlu/workspace/mjlab/logs/rsl_rl/g1_recovery/2026-04-11_10-46-43`

Observed behavior:

- reset artifacts were reduced
- standing-related rewards rose substantially
- but later checkpoints still drifted toward high-variance, unstable behavior

End-of-run summary:

- `Train/mean_reward`: `32.03`
- `Train/mean_episode_length`: `213.37`
- `Policy/mean_std`: `2.10`
- `Episode_Reward/stand_bonus`: `0.65`
- `Episode_Termination/joint_vel_limit`: `16.33`

Interpretation:

- reset direction was better
- optimizer / exploration remained too aggressive

### 3. New reset + conservative PPO

Run:

- `/home/ydlu/workspace/mjlab/logs/rsl_rl/g1_recovery/2026-04-11_19-29-55`

Observed behavior:

- substantially healthier learning dynamics
- policy variance stayed controlled
- recovery rewards and stand bonus grew strongly
- joint-velocity-limit termination stayed near zero

End-of-run summary:

- `Train/mean_reward`: `92.60`
- `Train/mean_episode_length`: `359.63`
- `Policy/mean_std`: `0.70`
- `Episode_Reward/stand_bonus`: `1.10`
- `Episode_Termination/joint_vel_limit`: `0.13`

Interpretation:

- this is the current best non-assisted recovery baseline
- unlike the earlier runs, it does not obviously drift into "high-std shaking"

### Current best checkpoint candidates

For the conservative run, the most useful checkpoints to inspect are:

- `model_3000.pt`
- `model_5000.pt`
- `model_5999.pt`

Recommended first checkpoint to inspect:

- `model_5000.pt`

Reason:

- it sits near the strongest reward / episode-length region
- policy variance remains much better controlled than in the older runs

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

### 5. Preview motion-derived fallen reset states

```bash
cd /home/ydlu/workspace/mjlab

uv run python -m mjlab.scripts.preview_recovery_fallen_poses \
  --motion-path "/home/ydlu/workspace/mjlab/artifacts/recovery_motions/g1_amp_get_up" \
  --settle-steps 5
```

This is the preferred way to inspect whether the reset distribution is
physically plausible before launching training.

To render the actual settle process, use:

```bash
cd /home/ydlu/workspace/mjlab

uv run python -m mjlab.scripts.preview_recovery_fallen_poses \
  --motion-path "/home/ydlu/workspace/mjlab/artifacts/recovery_motions/g1_amp_get_up" \
  --num-samples 3 \
  --progress-max 0.08 \
  --settle-steps 10 \
  --show-settle-steps \
  --settle-step-delay 0.05 \
  --seconds-per-pose 1.5
```

### 6. Evaluate a checkpoint under motion-derived fallen resets

```bash
cd /home/ydlu/workspace/mjlab

uv run play Mjlab-Recovery-Flat-Unitree-G1 \
  --checkpoint-file "/home/ydlu/workspace/mjlab/logs/rsl_rl/g1_recovery/2026-04-11_19-29-55/model_5000.pt" \
  --num-envs 1 \
  --viewer native \
  --no-terminations True \
  --teacher-motion-path "/home/ydlu/workspace/mjlab/artifacts/recovery_motions/g1_amp_get_up"
```

## Recommended Next Step

Because the previously launched longer recovery runs were started before the
reset improvements landed, they should be treated as diagnostic runs rather than
the main training line.

The recommended next step is:

1. Keep the successful motion-derived reset preview as the current validation
   baseline.
2. Launch a fresh short training run using the new reset strategy.
3. Compare early checkpoints under motion-derived fallen reset evaluation.
4. Only after that start a longer `4096`-environment run.

Suggested fresh smoke run:

```bash
cd /home/ydlu/workspace/mjlab

uv run train Mjlab-Recovery-Flat-Unitree-G1 \
  --env.teacher.motion-path "/home/ydlu/workspace/mjlab/artifacts/recovery_motions/g1_amp_get_up" \
  --env.scene.num-envs 512 \
  --agent.max-iterations 300
```

If this run shows healthier qualitative recovery than the earlier
random-air-drop-biased runs, it should replace them as the new baseline.

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

## Progressive Training Playbook

The recommended path is to train recovery in stages rather than jumping
straight to a large final run or to AMP.

### Stage 0: Data sanity

Before training, confirm that converted get-up motions play correctly:

- use `play_recovery_motion.py`
- confirm no global body twisting, no floating, and no obvious frame flips
- if playback is wrong, fix conversion before spending time on RL

### Stage 1: Smoke-test training

Goal:

- verify that reset, rewards, curriculum, and logging all run without failure

Recommended command:

```bash
cd /home/ydlu/workspace/mjlab

uv run train Mjlab-Recovery-Flat-Unitree-G1 \
  --env.teacher.motion-path "/home/ydlu/workspace/mjlab/artifacts/recovery_motions/g1_amp_get_up" \
  --env.scene.num-envs 128 \
  --agent.max-iterations 20
```

Success criteria:

- no NaNs
- no reset/runtime crashes
- non-zero `base_height_progress`
- non-zero `target_base_height` or `stand_bonus` emerging

### Stage 2: Medium-scale validation run

Goal:

- check whether the task genuinely learns before launching a long run

Recommended command:

```bash
cd /home/ydlu/workspace/mjlab

uv run train Mjlab-Recovery-Flat-Unitree-G1 \
  --env.teacher.motion-path "/home/ydlu/workspace/mjlab/artifacts/recovery_motions/g1_amp_get_up" \
  --env.scene.num-envs 512 \
  --agent.max-iterations 200
```

What to watch:

- `Train/mean_episode_length`
- `Episode_Reward/base_height_progress`
- `Episode_Reward/target_base_height`
- `Episode_Reward/stand_bonus`
- `Episode_Termination/root_height_floor`

Interpretation:

- if episode length and standing-related rewards rise, the task is learnable
- if `root_height_floor` dominates throughout, reset difficulty or termination
  thresholds may still be too harsh

### Stage 3: Stabilize functional recovery

Goal:

- make the robot reliably stand up, even if the motion is not yet very natural

Recommended approach:

- keep the current non-AMP formulation
- use teacher resets as a learning aid, not as a policy input
- tune curriculum before changing algorithms

Priority knobs:

- `teacher.reference_probability`
- random-fall reset difficulty
- `root_height_floor` threshold
- push disturbance magnitude
- reward weights for standing and stabilization

Practical rule:

- first optimize for "can stand up reliably"
- only afterward optimize for "looks natural"

### Stage 4: Decay teacher support

Goal:

- shift the policy from teacher-assisted initialization toward arbitrary-fall
  recovery

Recommended approach:

- reduce `teacher.reference_probability` over training
- widen random fall distributions
- increase perturbation difficulty only after standing becomes stable

This is already partially encoded in the current curriculum:

- early training uses more teacher reset states
- later training reduces teacher reset probability
- push strength can be increased in stages

### Stage 5: Evaluate whether AMP is actually needed

Only consider AMP after you have a strong non-AMP recovery baseline.

Signs that you may need AMP or another style prior:

- the policy can stand up but does so with visibly awkward or jerky motions
- reward curves look good while qualitative behavior remains unnatural
- posture/smoothness penalties are not enough to improve motion quality

Signs that you should **not** jump to AMP yet:

- the policy still frequently fails to stand
- reset/termination settings are still unstable
- you do not yet know what the best non-AMP baseline looks like

## Interpreting `amp_rec` Reference Logs

The original `amp_rec` AMP run is valuable as a reference, but mainly for
qualitative stability reasoning rather than direct scalar matching.

Useful takeaways from the original logs:

- early training can improve substantially before later behavior degrades
- scalar stability does not guarantee motion quality stability
- intermediate checkpoints may be better than final checkpoints

Practical rule:

- do not assume the final checkpoint is the best checkpoint
- periodically play intermediate `mjlab` checkpoints during training
- compare behavior quality, not just reward curves

## Current Limitations

- AMP discriminator logic has not been ported into `mjlab`
- the current converted motion format is aimed at recovery reset/playback, not
  the full `mjlab` tracking pipeline
- the playback tool is intended for quick qualitative inspection, not for exact
  tracking-metric evaluation
