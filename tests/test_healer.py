"""Tests for K8s self-healer."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
from healer import PodStatus, HealAction, HealResult


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
