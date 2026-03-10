Motion tracking with custom datasets
====================================

This page walks through how to use a custom motion dataset (for example, a CSV
export from an external retargeting pipeline) with mjlab's tracking tasks. The
workflow has three main steps:

1. Convert your CSV into mjlab's standard ``motion.npz`` format.
2. Upload the motion to a Weights & Biases (W&B) registry.
3. Visualize and train a tracking policy in mjlab using that motion.

The examples below assume a Unitree G1 tracking task, but the same pattern
applies to other robots and datasets.


1. Converting CSV to ``motion.npz``
-----------------------------------

If you have a CSV file containing base pose and joint positions, you can
convert it to mjlab's standard format using the ``csv_to_npz`` script.

.. code-block:: bash

   uv run python -m mjlab.scripts.csv_to_npz \
     --input-file /home/ydlu/workspace/whole_body_tracking/LAFAN1_Retargeting_Dataset/g1/fallAndGetUp2_subject2.csv \
     --output-name fallAndGetUp2_subject2_g1mj \
     --input-fps 30 \
     --output-fps 50 \
     --render True

Arguments:

- **``--input-file``**: Path to your CSV file. The example above uses a
  LAFAN1-based retargeted dataset for Unitree G1.
- **``--output-name``**: Logical name for this motion sequence. This becomes
  both the local identifier and the W&B artifact name.
- **``--input-fps``**: Frame rate of the original CSV.
- **``--output-fps``**: Desired frame rate in the simulator.
- **``--render``**: When ``True``, the script renders a video and logs it to
  W&B alongside the motion.

What the script does
~~~~~~~~~~~~~~~~~~~~

The script performs several operations:

- Loads your CSV into tensors (base position, base orientation, joint
  positions).
- Interpolates the motion from ``input_fps`` to ``output_fps``.
- Computes base and joint velocities.
- Maps the CSV joint ordering to mjlab's internal Unitree G1 joint ordering
  using the configured ``joint_names`` list.
- Simulates the robot with the motion applied to verify consistency.
- Saves the result as ``/tmp/motion.npz`` with the following fields:

  - ``joint_pos``
  - ``joint_vel``
  - ``body_pos_w``
  - ``body_quat_w``
  - ``body_lin_vel_w``
  - ``body_ang_vel_w``

- Uploads the file to W&B as an artifact of type ``motions`` under the
  collection name you specified with ``--output-name``.

Numerical checks
~~~~~~~~~~~~~~~~

During conversion, the script checks that analytically-computed base velocities
match the simulated velocities up to a small tolerance. Small numerical
differences (for example, on the order of :math:`10^{-5}`) are allowed and will
not cause the conversion to fail, but large discrepancies will still surface as
errors.


2. Verifying the motion in W&B
------------------------------

After a successful run, you should see console output similar to:

.. code-block:: text

   Saving to /tmp/motion.npz...
   Uploading to Weights & Biases...
   [INFO]: Motion saved to wandb registry: motions/fallAndGetUp2_subject2_g1mj

This means the motion is now available as a W&B artifact with a registry path
like:

.. code-block:: text

   wandb-registry-motions/fallAndGetUp2_subject2_g1mj

If ``--render True`` was enabled, the script also:

- Saves a local ``motion.mp4`` in the current directory.
- Logs the video to W&B so you can quickly inspect the trajectory in the
  browser.


3. Visualizing the motion in mjlab
----------------------------------

Before training a tracking policy, it is often useful to verify that the motion
plays back correctly in mjlab. You can do this using the ``play`` script with a
dummy agent (no trained policy required):

.. code-block:: bash

   uv run play Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation \
     --agent zero \
     --registry-name teampeterchong-org/wandb-registry-motions/fallAndGetUp2_subject2_g1mj \
     --num-envs 1 \
     --viewer auto \
     --no-terminations True

Here:

- ``--agent zero`` uses a zero-action dummy agent so you are only viewing the
  reference motion.
- ``--registry-name`` points to the W&B motion artifact created in the previous
  step.
- ``--num-envs 1`` keeps visualization simple for inspection.
- ``--viewer auto`` selects a native MuJoCo window when a display is available,
  and falls back to a web-based Viser viewer otherwise.
- ``--no-terminations True`` disables termination conditions so the motion can
  be viewed without early resets.


4. Training a tracking policy on the motion
-------------------------------------------

Once you are satisfied with the motion playback, you can use the same registry
name to train a tracking policy with RSL-RL:

.. code-block:: bash

   uv run train Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation \
     --registry-name teampeterchong-org/wandb-registry-motions/fallAndGetUp2_subject2_g1mj \
     --env.scene.num-envs 4096 \
     --agent.wandb_project rec \
     --agent.run_name g1_fallAndGetUp2_subject2_mj \
     --gpu-ids [0]

Notes:

- ``--registry-name`` must match the W&B motion registry path you created
  earlier.
- ``--env.scene.num-envs`` controls the number of parallel environments (and
  thus throughput).
- ``--agent.wandb_project`` sets the W&B project name for logging training
  metrics.
- ``--agent.run_name`` sets a readable label for this training run; the short
  run ID still appears in the URL and is used when resuming.
- ``--gpu-ids`` selects which visible GPUs to use (for example, ``[0]``,
  ``[0,1]``, or ``all``).


5. Playing back a trained policy
--------------------------------

There are two common ways to visualize a trained policy:


5.1 From a W&B run
~~~~~~~~~~~~~~~~~~

After training, you can point ``play`` at the W&B run that contains your
checkpoint:

.. code-block:: bash

   # Suppose your training run URL is:
   # https://wandb.ai/teampeterchong/rec/runs/kdvolllo
   # Then wandb_run_path is: teampeterchong/rec/kdvolllo

   uv run play Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation \
     --agent trained \
     --wandb-run-path teampeterchong/rec/kdvolllo \
     --viewer auto \
     --num-envs 1

In this mode, ``play`` will:

- Resolve the latest checkpoint for that run (downloading from W&B if needed).
- Load the motion artifact used during training.
- Launch a viewer (native or web-based) to roll out the policy.


5.2 From a local checkpoint and motion file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Alternatively, you can play back a specific local checkpoint together with a
local ``motion.npz`` file. This is useful when you have copied artifacts out of
W&B (for example into an ``artifacts/`` directory) or when W&B is not
available.

.. code-block:: bash

   uv run play Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation \
     --agent trained \
     --checkpoint-file logs/rsl_rl/g1_tracking/2026-03-10_11-44-32_g1_fallAndGetUp2_subject2_mj/model_500.pt \
     --motion-file artifacts/fallAndGetUp2_subject2_g1mj:v0/motion.npz \
     --viewer viser \
     --num-envs 1

Notes:

- For tracking tasks, ``play`` requires a motion source when you pass a
  ``checkpoint_file``. You must provide either:

  - ``--motion-file /path/to/motion.npz`` (local file), or
  - ``--wandb-run-path entity/project/run_id`` so that the motion artifact can
    be resolved from W&B.

- ``--viewer viser`` starts a Viser web viewer. When running on a remote
  server, you can use SSH port forwarding or similar techniques to open the
  viewer URL in a local browser.

