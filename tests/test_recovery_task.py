"""Tests specific to recovery tasks."""

import pytest

from mjlab.asset_zoo.robots import G1_ACTION_SCALE
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.tasks.recovery.recovery_env_cfg import RecoveryEnvCfg


@pytest.fixture(scope="module")
def recovery_task_ids() -> list[str]:
    """Get all recovery task IDs."""
    return [t for t in list_tasks() if "Recovery" in t]


def test_recovery_tasks_are_registered(recovery_task_ids: list[str]) -> None:
    """At least one recovery task should be registered."""
    assert recovery_task_ids, "Expected at least one registered recovery task"


def test_recovery_tasks_do_not_expose_reference_commands(
    recovery_task_ids: list[str],
) -> None:
    """Recovery tasks should not depend on reference-motion commands."""
    for task_id in recovery_task_ids:
        cfg = load_env_cfg(task_id)
        assert cfg.commands == {}, (
            f"Task {task_id} exposes commands={cfg.commands}, "
            "expected none for recovery"
        )


def test_g1_recovery_has_required_sensors(recovery_task_ids: list[str]) -> None:
    """G1 recovery tasks should retain collision/contact sensing."""
    for task_id in recovery_task_ids:
        cfg = load_env_cfg(task_id)
        assert cfg.scene.sensors is not None, (
            f"Task {task_id} has no sensors"
        )
        sensor_names = {s.name for s in cfg.scene.sensors}
        assert "feet_ground_contact" in sensor_names, (
            f"Task {task_id} missing feet_ground_contact sensor"
        )
        assert "self_collision" in sensor_names, (
            f"Task {task_id} missing self_collision sensor"
        )


def test_g1_recovery_has_correct_action_scale(recovery_task_ids: list[str]) -> None:
    """G1 recovery tasks should use G1_ACTION_SCALE."""
    for task_id in recovery_task_ids:
        cfg = load_env_cfg(task_id)
        joint_pos_action = cfg.actions["joint_pos"]
        assert isinstance(joint_pos_action, JointPositionActionCfg), (
            f"Task {task_id} joint_pos action is not JointPositionActionCfg"
        )
        assert joint_pos_action.scale == G1_ACTION_SCALE, (
            "Task "
            f"{task_id} action scale mismatch, expected G1_ACTION_SCALE"
        )


def test_recovery_play_mode_disables_teacher_and_push(
    recovery_task_ids: list[str],
) -> None:
    """Play mode should remove deployment-time teacher dependencies."""
    for task_id in recovery_task_ids:
        cfg = load_env_cfg(task_id, play=True)
        assert isinstance(cfg, RecoveryEnvCfg), (
            f"Task {task_id} did not load as RecoveryEnvCfg in play mode"
        )
        assert "push_robot" not in cfg.events, (
            f"Task {task_id} (play mode) still has push_robot enabled"
        )
        assert "upward_assist" not in cfg.events, (
            f"Task {task_id} (play mode) still has upward_assist enabled"
        )
        assert cfg.teacher.motion_path == "", (
            f"Task {task_id} (play mode) kept teacher motion_path="
            f"{cfg.teacher.motion_path}"
        )
        assert cfg.teacher.reference_probability == 0.0, (
            f"Task {task_id} (play mode) kept teacher probability "
            f"{cfg.teacher.reference_probability}"
        )
        for startup_event in ["foot_friction", "encoder_bias", "base_com"]:
            assert startup_event not in cfg.events, (
                f"Task {task_id} (play mode) still has {startup_event} enabled"
            )
        reset_params = cfg.events["recovery_reset"].params
        assert reset_params["fallen_pose_probability"] == 1.0, (
            "Task "
            f"{task_id} (play mode) should use only fallen poses for reset"
        )
        assert reset_params["random_fall_probability"] == 0.0, (
            f"Task {task_id} (play mode) should disable random air-drop reset"
        )
        assert reset_params["fallen_pose_yaw_jitter"] == 0.0, (
            f"Task {task_id} (play mode) should disable yaw jitter for reset"
        )
        assert reset_params["fallen_pose_joint_jitter"] == 0.0, (
            f"Task {task_id} (play mode) should disable joint jitter for reset"
        )
        assert reset_params["teacher_post_reset_settle_steps"] == 0, (
            f"Task {task_id} (play mode) should not settle teacher resets"
        )
        assert reset_params["fallen_post_reset_settle_steps"] == 12, (
            "Task "
            f"{task_id} (play mode) should use longer settle for fallen resets"
        )
        assert reset_params["random_post_reset_settle_steps"] == 12, (
            "Task "
            f"{task_id} (play mode) should use longer settle for random fallback resets"
        )


def test_recovery_actor_observations_are_motion_free(
    recovery_task_ids: list[str],
) -> None:
    """Actor observations should use body state only."""
    for task_id in recovery_task_ids:
        cfg = load_env_cfg(task_id)
        actor_terms = cfg.observations["actor"].terms
        forbidden_terms = {
            "command",
            "motion_anchor_pos_b",
            "motion_anchor_ori_b",
        }
        assert forbidden_terms.isdisjoint(actor_terms), (
            f"Task {task_id} actor observations unexpectedly contain "
            "reference-motion terms"
        )


def test_recovery_push_curriculum_strengthens_disturbances(
    recovery_task_ids: list[str],
) -> None:
    """Push curriculum should ramp disturbances over reachable stages."""
    for task_id in recovery_task_ids:
        cfg = load_env_cfg(task_id)
        stages = cfg.curriculum["push_velocity"].params["stages"]

        assert len(stages) == 4, (
            f"Task {task_id} should expose four push stages"
        )

        steps = [stage["step"] for stage in stages]
        assert steps == sorted(steps), (
            f"Task {task_id} push stages must be ordered"
        )

        x_magnitudes = [
            abs(stage["velocity_range"]["x"][1]) for stage in stages
        ]
        y_magnitudes = [
            abs(stage["velocity_range"]["y"][1]) for stage in stages
        ]
        yaw_magnitudes = [
            abs(stage["velocity_range"]["yaw"][1]) for stage in stages
        ]

        assert x_magnitudes == sorted(x_magnitudes), (
            f"Task {task_id} x push magnitudes must be non-decreasing"
        )
        assert y_magnitudes == sorted(y_magnitudes), (
            f"Task {task_id} y push magnitudes must be non-decreasing"
        )
        assert yaw_magnitudes == sorted(yaw_magnitudes), (
            f"Task {task_id} yaw push magnitudes must be non-decreasing"
        )

        assert x_magnitudes[-1] > x_magnitudes[0], (
            "Task "
            f"{task_id} final x push stage should be stronger than the initial stage"
        )
        assert yaw_magnitudes[-1] > yaw_magnitudes[0], (
            "Task "
            f"{task_id} final yaw push stage should be stronger than the initial stage"
        )
