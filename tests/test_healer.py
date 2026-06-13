"""Tests for K8s self-healer."""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
import time
from healer import (
    PodStatus, HealAction, HealResult, HealDecision,
    heal_pod, _CooldownTracker, _execute_recommendation,
)
from llm import HealRecommendation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pod(
    name: str = "test-pod",
    namespace: str = "default",
    restart_count: int = 5,
    phase: str = "CrashLoopBackOff",
    container_statuses: list | None = None,
    events: list | None = None,
) -> PodStatus:
    if container_statuses is None:
        container_statuses = [
            {"state": {"waiting": {"reason": "CrashLoopBackOff"}}, "ready": False}
        ]
    return PodStatus(
        name=name,
        namespace=namespace,
        phase=phase,
        ready=False,
        restart_count=restart_count,
        container_statuses=container_statuses,
        events=events or [],
    )


class _FakeLLMClient:
    """A fake LLM client that returns a preset recommendation."""

    def __init__(self, action: str = "restart", confidence: float = 0.9,
                 reasoning: str = "test reasoning"):
        self._rec = HealRecommendation(
            action=action, confidence=confidence, reasoning=reasoning,
        )
        self.calls: list[dict] = []

    def recommend_action(self, pod_name, namespace, restart_count,
                         last_termination_reason, recent_events):
        self.calls.append({
            "pod_name": pod_name,
            "namespace": namespace,
            "restart_count": restart_count,
            "last_termination_reason": last_termination_reason,
            "recent_events": recent_events,
        })
        return self._rec


# ---------------------------------------------------------------------------
# Original dataclass tests
# ---------------------------------------------------------------------------

class TestPodStatus:
    def test_defaults(self):
        pod = PodStatus(
            name="test-pod",
            namespace="default",
            phase="CrashLoopBackOff",
            ready=False,
            restart_count=5
        )
        assert pod.name == "test-pod"
        assert pod.phase == "CrashLoopBackOff"
        assert not pod.ready
        assert pod.restart_count == 5


class TestHealAction:
    def test_defaults(self):
        action = HealAction(
            pod_name="test-pod",
            namespace="default",
            action="restart",
            reason="CrashLoopBackOff"
        )
        assert not action.success
        assert action.message == ""


class TestHealResult:
    def test_success(self):
        result = HealResult(
            pod_name="test-pod",
            namespace="default",
            original_status="CrashLoopBackOff",
            action_taken="restart",
            success=True,
            message="Pod restarted",
            new_status="Running"
        )
        assert result.success
        assert result.new_status == "Running"


# ---------------------------------------------------------------------------
# HealDecision
# ---------------------------------------------------------------------------

class TestHealDecision:
    def test_fields(self):
        d = HealDecision(
            recommended_action="restart",
            confidence=0.85,
            reasoning="CrashLoopBackOff with low restart count",
            source="llm",
        )
        assert d.recommended_action == "restart"
        assert d.confidence == 0.85
        assert d.source == "llm"

    def test_threshold_source(self):
        d = HealDecision(
            recommended_action="rollback",
            confidence=1.0,
            reasoning="restart_count > 10",
            source="threshold",
        )
        assert d.source == "threshold"


# ---------------------------------------------------------------------------
# LLM-guided healing
# ---------------------------------------------------------------------------

class TestHealPodWithLLM:
    """Tests for heal_pod() when an LLM client is provided."""

    @patch("healer.restart_pod")
    def test_llm_restart_recommendation(self, mock_restart):
        mock_restart.return_value = HealAction(
            pod_name="test-pod", namespace="default",
            action="restart", reason="", success=True, message="deleted",
        )
        client = _FakeLLMClient(action="restart", confidence=0.9)
        pod = _make_pod(restart_count=2)

        action = heal_pod(pod, llm_client=client)

        assert action.action == "restart"
        assert action.success is True
        assert len(client.calls) == 1
        assert client.calls[0]["pod_name"] == "test-pod"
        mock_restart.assert_called_once()

    @patch("healer.rollback_deployment")
    @patch("healer.get_deployment_for_pod", return_value="my-deploy")
    def test_llm_rollback_recommendation(self, mock_get_deploy, mock_rollback):
        mock_rollback.return_value = HealAction(
            pod_name="my-deploy", namespace="default",
            action="rollback", reason="", success=True, message="rolled back",
        )
        client = _FakeLLMClient(action="rollback", confidence=0.85)
        pod = _make_pod(restart_count=8)

        action = heal_pod(pod, llm_client=client)

        assert action.action == "rollback"
        mock_rollback.assert_called_once_with("my-deploy", "default")

    @patch("healer.restart_pod")
    def test_llm_low_confidence_falls_back_to_threshold(self, mock_restart):
        """When confidence < 0.5, the threshold logic should take over."""
        mock_restart.return_value = HealAction(
            pod_name="test-pod", namespace="default",
            action="restart", reason="", success=True, message="deleted",
        )
        client = _FakeLLMClient(action="restart", confidence=0.3)
        pod = _make_pod(restart_count=5)

        action = heal_pod(pod, llm_client=client)

        # Threshold says restart (> 3 restarts)
        assert action.action == "restart"
        # The reason should still reflect the threshold source
        assert "[threshold]" in action.reason

    def test_llm_none_action_falls_back(self):
        """When LLM says 'none', threshold logic should decide."""
        client = _FakeLLMClient(action="none", confidence=0.9)
        pod = _make_pod(restart_count=1)  # below threshold

        action = heal_pod(pod, llm_client=client)

        assert action.action == "none"
        assert "[threshold]" in action.reason

    @patch("healer.restart_pod")
    def test_llm_alert_action_does_not_act(self, mock_restart):
        """When LLM says 'alert', no destructive action is taken."""
        client = _FakeLLMClient(action="alert", confidence=0.8)
        pod = _make_pod(restart_count=2)

        action = heal_pod(pod, llm_client=client)

        assert action.action == "alert"
        mock_restart.assert_not_called()

    @patch("healer.restart_pod")
    def test_llm_sends_correct_context(self, mock_restart):
        """Verify the LLM client receives the right pod context."""
        mock_restart.return_value = HealAction(
            pod_name="test-pod", namespace="ns1",
            action="restart", reason="", success=True, message="ok",
        )
        client = _FakeLLMClient(action="restart", confidence=0.9)
        container_statuses = [
            {"state": {"terminated": {"reason": "OOMKilled"}}, "ready": False}
        ]
        events = [{"type": "Warning", "reason": "BackOff", "message": "Back-off restarting"}]
        pod = _make_pod(
            name="test-pod", namespace="ns1", restart_count=4,
            container_statuses=container_statuses, events=events,
        )

        heal_pod(pod, llm_client=client)

        call = client.calls[0]
        assert call["pod_name"] == "test-pod"
        assert call["namespace"] == "ns1"
        assert call["restart_count"] == 4
        assert call["last_termination_reason"] == "OOMKilled"
        assert call["recent_events"] == events

    @patch("healer.restart_pod")
    def test_llm_error_falls_back_to_threshold(self, mock_restart):
        """When the LLM raises an exception, threshold logic takes over."""
        mock_restart.return_value = HealAction(
            pod_name="test-pod", namespace="default",
            action="restart", reason="", success=True, message="deleted",
        )
        client = MagicMock()
        client.recommend_action.side_effect = ConnectionError("no LLM")
        pod = _make_pod(restart_count=5)

        action = heal_pod(pod, llm_client=client)

        assert action.action == "restart"
        assert "[threshold]" in action.reason
        mock_restart.assert_called_once()


# ---------------------------------------------------------------------------
# Threshold-based fallback (no LLM)
# ---------------------------------------------------------------------------

class TestHealPodThreshold:
    """Tests for threshold-based healing when no LLM client is provided."""

    @patch("healer.rollback_deployment")
    @patch("healer.get_deployment_for_pod", return_value="my-deploy")
    def test_high_restart_count_triggers_rollback(self, mock_deploy, mock_rollback):
        mock_rollback.return_value = HealAction(
            pod_name="my-deploy", namespace="default",
            action="rollback", reason="", success=True, message="rolled back",
        )
        pod = _make_pod(restart_count=11)

        action = heal_pod(pod)

        assert action.action == "rollback"
        assert "[threshold]" in action.reason

    @patch("healer.restart_pod")
    def test_moderate_restart_count_triggers_restart(self, mock_restart):
        mock_restart.return_value = HealAction(
            pod_name="test-pod", namespace="default",
            action="restart", reason="", success=True, message="deleted",
        )
        pod = _make_pod(restart_count=4)

        action = heal_pod(pod)

        assert action.action == "restart"
        assert "[threshold]" in action.reason

    def test_low_restart_count_no_action(self):
        pod = _make_pod(restart_count=2)

        action = heal_pod(pod)

        assert action.action == "none"
        assert "not in critical state" in action.reason.lower() or "threshold" in action.reason

    @patch("healer.restart_pod")
    def test_force_restart_overrides_threshold(self, mock_restart):
        mock_restart.return_value = HealAction(
            pod_name="test-pod", namespace="default",
            action="restart", reason="", success=True, message="deleted",
        )
        pod = _make_pod(restart_count=1)

        action = heal_pod(pod, force_restart=True)

        assert action.action == "restart"

    @patch("healer.restart_pod")
    @patch("healer.get_deployment_for_pod", return_value=None)
    def test_high_restart_without_deploy_restarts_instead(self, mock_deploy, mock_restart):
        """When restarts > 10 but there is no deployment, restart instead of rollback."""
        mock_restart.return_value = HealAction(
            pod_name="test-pod", namespace="default",
            action="restart", reason="", success=True, message="deleted",
        )
        pod = _make_pod(restart_count=11)

        action = heal_pod(pod)

        assert action.action == "restart"


# ---------------------------------------------------------------------------
# Cooldown tracker
# ---------------------------------------------------------------------------

class TestCooldownTracker:
    def test_initial_allows_heal(self):
        tracker = _CooldownTracker(cooldown_seconds=60, max_heals_per_pod=3)
        can, reason = tracker.can_heal("pod-1", "default")
        assert can is True
        assert reason == ""

    def test_cooldown_blocks_repeated_heals(self):
        tracker = _CooldownTracker(cooldown_seconds=60, max_heals_per_pod=3)
        tracker.record_heal("pod-1", "default")

        can, reason = tracker.can_heal("pod-1", "default")
        assert can is False
        assert "cooldown" in reason.lower()

    def test_cooldown_expires(self):
        tracker = _CooldownTracker(cooldown_seconds=1, max_heals_per_pod=3)
        tracker.record_heal("pod-1", "default")

        # Should be blocked immediately
        can, _ = tracker.can_heal("pod-1", "default")
        assert can is False

        # Wait for cooldown to expire
        time.sleep(1.1)
        can, reason = tracker.can_heal("pod-1", "default")
        assert can is True
        assert reason == ""

    def test_max_heals_per_pod_blocks(self):
        tracker = _CooldownTracker(cooldown_seconds=0, max_heals_per_pod=2)

        tracker.record_heal("pod-1", "default")
        tracker.record_heal("pod-1", "default")

        can, reason = tracker.can_heal("pod-1", "default")
        assert can is False
        assert "max heal limit" in reason.lower()

    def test_different_pods_independent(self):
        tracker = _CooldownTracker(cooldown_seconds=60, max_heals_per_pod=2)

        tracker.record_heal("pod-1", "default")
        tracker.record_heal("pod-1", "default")

        # pod-1 is at max, pod-2 is fine
        can1, _ = tracker.can_heal("pod-1", "default")
        can2, _ = tracker.can_heal("pod-2", "default")
        assert can1 is False
        assert can2 is True

    def test_heal_count(self):
        tracker = _CooldownTracker(cooldown_seconds=60, max_heals_per_pod=3)
        assert tracker.heal_count("pod-1", "default") == 0

        tracker.record_heal("pod-1", "default")
        assert tracker.heal_count("pod-1", "default") == 1

        tracker.record_heal("pod-1", "default")
        assert tracker.heal_count("pod-1", "default") == 2

    def test_namespaced_pods_independent(self):
        tracker = _CooldownTracker(cooldown_seconds=0, max_heals_per_pod=1)
        tracker.record_heal("pod-1", "ns-a")

        can_a, _ = tracker.can_heal("pod-1", "ns-a")
        can_b, _ = tracker.can_heal("pod-1", "ns-b")
        assert can_a is False
        assert can_b is True


# ---------------------------------------------------------------------------
# _execute_recommendation
# ---------------------------------------------------------------------------

class TestExecuteRecommendation:
    @patch("healer.restart_pod")
    def test_restart(self, mock_restart):
        mock_restart.return_value = HealAction(
            pod_name="p", namespace="d", action="restart", reason="", success=True, message="ok",
        )
        pod = _make_pod()
        result = _execute_recommendation(pod, "restart")
        assert result.action == "restart"

    @patch("healer.rollback_deployment")
    @patch("healer.get_deployment_for_pod", return_value="my-deploy")
    def test_rollback(self, mock_deploy, mock_rollback):
        mock_rollback.return_value = HealAction(
            pod_name="my-deploy", namespace="d", action="rollback", reason="", success=True, message="ok",
        )
        pod = _make_pod()
        result = _execute_recommendation(pod, "rollback")
        assert result.action == "rollback"

    def test_alert(self):
        pod = _make_pod()
        result = _execute_recommendation(pod, "alert")
        assert result.action == "alert"
        assert not result.success


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
