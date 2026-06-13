"""Kubernetes self-healing agent for detecting and fixing problematic pods."""

import subprocess
import json
import time
from dataclasses import dataclass, field
from typing import Optional, Protocol


@dataclass
class PodStatus:
    """Status of a Kubernetes pod."""
    name: str
    namespace: str
    phase: str
    ready: bool
    restart_count: int
    container_statuses: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)


@dataclass
class HealAction:
    """A healing action to take."""
    pod_name: str
    namespace: str
    action: str  # restart, rollback, scale, delete
    reason: str
    success: bool = False
    message: str = ""


@dataclass
class HealResult:
    """Result of a healing operation."""
    pod_name: str
    namespace: str
    original_status: str
    action_taken: str
    success: bool
    message: str
    new_status: str = ""


@dataclass
class HealDecision:
    """Records how a healing decision was made."""
    recommended_action: str  # restart, rollback, scale, alert, none
    confidence: float  # 0.0 to 1.0
    reasoning: str
    source: str  # "llm" or "threshold"


class LLMClient(Protocol):
    """Protocol for LLM clients that can recommend healing actions."""

    def recommend_action(
        self,
        pod_name: str,
        namespace: str,
        restart_count: int,
        last_termination_reason: str,
        recent_events: list[dict],
    ) -> "HealRecommendation":
        ...


class _CooldownTracker:
    """Tracks per-pod cooldown and heal counts."""

    def __init__(self, cooldown_seconds: int = 300, max_heals_per_pod: int = 3):
        self.cooldown_seconds = cooldown_seconds
        self.max_heals_per_pod = max_heals_per_pod
        self._last_heal: dict[str, float] = {}  # pod_key -> timestamp
        self._heal_counts: dict[str, int] = {}  # pod_key -> count

    def _key(self, pod_name: str, namespace: str) -> str:
        return f"{namespace}/{pod_name}"

    def can_heal(self, pod_name: str, namespace: str) -> tuple[bool, str]:
        """Check if a pod is eligible for healing. Returns (allowed, reason)."""
        key = self._key(pod_name, namespace)

        # Check max heals
        count = self._heal_counts.get(key, 0)
        if count >= self.max_heals_per_pod:
            return False, f"Pod {key} has reached max heal limit ({self.max_heals_per_pod})"

        # Check cooldown
        last = self._last_heal.get(key)
        if last is not None:
            elapsed = time.time() - last
            if elapsed < self.cooldown_seconds:
                remaining = int(self.cooldown_seconds - elapsed)
                return False, f"Pod {key} in cooldown ({remaining}s remaining)"

        return True, ""

    def record_heal(self, pod_name: str, namespace: str) -> None:
        """Record that a heal was performed on a pod."""
        key = self._key(pod_name, namespace)
        self._last_heal[key] = time.time()
        self._heal_counts[key] = self._heal_counts.get(key, 0) + 1

    def heal_count(self, pod_name: str, namespace: str) -> int:
        """Get the number of times a pod has been healed."""
        return self._heal_counts.get(self._key(pod_name, namespace), 0)


def get_problematic_pods(namespace: str = "default") -> list[PodStatus]:
    """Get pods with issues."""
    cmd = ["kubectl", "get", "pods", "-n", namespace, "-o", "json"]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        return []

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return []

    problematic = []
    for pod in data.get("items", []):
        phase = pod.get("status", {}).get("phase", "")
        name = pod.get("metadata", {}).get("name", "")
        ns = pod.get("metadata", {}).get("namespace", namespace)

        # Check container statuses
        container_statuses = pod.get("status", {}).get("containerStatuses", [])
        restart_count = sum(cs.get("restartCount", 0) for cs in container_statuses)

        # Determine if pod is problematic
        is_problematic = False
        for cs in container_statuses:
            state = cs.get("state", {})
            if "waiting" in state:
                reason = state["waiting"].get("reason", "")
                if reason in ("CrashLoopBackOff", "ImagePullBackOff",
                             "ErrImagePull", "CreateContainerConfigError"):
                    is_problematic = True
                    break
            if "terminated" in state:
                reason = state["terminated"].get("reason", "")
                if reason == "Error":
                    is_problematic = True
                    break

        # Check for high restart count
        if restart_count > 5:
            is_problematic = True

        if is_problematic:
            status = PodStatus(
                name=name,
                namespace=ns,
                phase=phase,
                ready=all(cs.get("ready", False) for cs in container_statuses),
                restart_count=restart_count,
                container_statuses=container_statuses,
            )
            problematic.append(status)

    return problematic


def get_pod_events(pod_name: str, namespace: str = "default") -> list[dict]:
    """Get events for a specific pod."""
    cmd = ["kubectl", "get", "events", "-n", namespace,
           "--field-selector", f"involvedObject.name={pod_name}", "-o", "json"]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        return []

    try:
        data = json.loads(result.stdout)
        return data.get("items", [])
    except json.JSONDecodeError:
        return []


def restart_pod(pod_name: str, namespace: str = "default") -> HealAction:
    """Restart a pod by deleting it (Deployment will recreate)."""
    action = HealAction(
        pod_name=pod_name,
        namespace=namespace,
        action="restart",
        reason="Pod in CrashLoopBackOff, attempting restart"
    )

    cmd = ["kubectl", "delete", "pod", pod_name, "-n", namespace]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        action.success = True
        action.message = f"Pod {pod_name} deleted successfully"
    else:
        action.message = f"Failed to delete pod: {result.stderr}"

    return action


def rollback_deployment(deployment_name: str, namespace: str = "default") -> HealAction:
    """Rollback a deployment to previous revision."""
    action = HealAction(
        pod_name=deployment_name,
        namespace=namespace,
        action="rollback",
        reason="Multiple pod failures, attempting rollback"
    )

    cmd = ["kubectl", "rollout", "undo", f"deployment/{deployment_name}", "-n", namespace]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        action.success = True
        action.message = f"Deployment {deployment_name} rolled back"
    else:
        action.message = f"Failed to rollback: {result.stderr}"

    return action


def scale_deployment(deployment_name: str, replicas: int,
                     namespace: str = "default") -> HealAction:
    """Scale a deployment."""
    action = HealAction(
        pod_name=deployment_name,
        namespace=namespace,
        action="scale",
        reason=f"Scaling to {replicas} replicas"
    )

    cmd = ["kubectl", "scale", f"deployment/{deployment_name}",
           f"--replicas={replicas}", "-n", namespace]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        action.success = True
        action.message = f"Deployment {deployment_name} scaled to {replicas}"
    else:
        action.message = f"Failed to scale: {result.stderr}"

    return action


def get_deployment_for_pod(pod_name: str, namespace: str = "default") -> Optional[str]:
    """Get the deployment name for a pod."""
    cmd = ["kubectl", "get", "pod", pod_name, "-n", namespace,
           "-o", "jsonpath={.metadata.ownerReferences[0].name}"]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        name = result.stdout.strip()
        return name if name else None

    return None


def heal_pod(
    pod: PodStatus,
    force_restart: bool = False,
    llm_client: Optional[object] = None,
) -> HealAction:
    """Attempt to heal a problematic pod.

    When *llm_client* is provided it is called first to get a recommendation.
    The LLM recommendation is used as the primary decision source, with
    threshold-based logic as a fallback when the LLM is unavailable or
    returns a low-confidence / unactionable result.
    """
    decision: Optional[HealDecision] = None
    action: Optional[HealAction] = None

    # --- Try LLM-guided decision first ---
    if llm_client is not None:
        try:
            # Extract last termination reason from container statuses
            last_reason = ""
            for cs in pod.container_statuses:
                state = cs.get("state", {})
                if "terminated" in state:
                    last_reason = state["terminated"].get("reason", "")
                elif "waiting" in state:
                    last_reason = state["waiting"].get("reason", "")

            events = pod.events or get_pod_events(pod.name, pod.namespace)
            rec = llm_client.recommend_action(
                pod_name=pod.name,
                namespace=pod.namespace,
                restart_count=pod.restart_count,
                last_termination_reason=last_reason,
                recent_events=events,
            )

            decision = HealDecision(
                recommended_action=rec.action,
                confidence=rec.confidence,
                reasoning=rec.reasoning,
                source="llm",
            )

            # Only follow LLM recommendation when confidence is reasonable
            if rec.confidence >= 0.5 and rec.action != "none":
                action = _execute_recommendation(pod, rec.action)

        except (ConnectionError, Exception):
            # LLM unavailable -- fall through to threshold logic
            pass

    # --- Threshold-based fallback ---
    if action is None:
        deployment = get_deployment_for_pod(pod.name, pod.namespace)

        if pod.restart_count > 10 and deployment:
            action = rollback_deployment(deployment, pod.namespace)
            fallback_action = "rollback"
        elif force_restart or pod.restart_count > 3:
            action = restart_pod(pod.name, pod.namespace)
            fallback_action = "restart"
        else:
            action = HealAction(
                pod_name=pod.name,
                namespace=pod.namespace,
                action="none",
                reason="Pod not in critical state yet",
            )
            fallback_action = "none"

        # Always overwrite decision with the threshold-based one when
        # the fallback path was taken (even if the LLM was consulted
        # but didn't produce an actionable recommendation).
        decision = HealDecision(
            recommended_action=fallback_action,
            confidence=1.0,
            reasoning=f"Threshold-based: restart_count={pod.restart_count}",
            source="threshold",
        )

    # Attach decision metadata to the action's reason
    action.reason = (
        f"[{decision.source}] {decision.recommended_action} "
        f"(conf={decision.confidence:.2f}): {decision.reasoning}"
    )

    return action


def _execute_recommendation(pod: PodStatus, action_name: str) -> HealAction:
    """Turn a string recommendation into an actual HealAction."""
    deployment = get_deployment_for_pod(pod.name, pod.namespace)

    if action_name == "rollback" and deployment:
        return rollback_deployment(deployment, pod.namespace)

    if action_name == "scale" and deployment:
        return scale_deployment(deployment, 1, pod.namespace)

    if action_name == "restart":
        return restart_pod(pod.name, pod.namespace)

    # "alert" or anything else -- don't act, just report
    return HealAction(
        pod_name=pod.name,
        namespace=pod.namespace,
        action="alert" if action_name == "alert" else action_name,
        reason="LLM recommends human review",
    )


def monitor_and_heal(
    namespace: str = "default",
    interval: int = 30,
    max_heals: int = 10,
    llm_client: Optional[object] = None,
    cooldown_seconds: int = 300,
    max_heals_per_pod: int = 3,
) -> list[HealResult]:
    """Monitor pods and heal problematic ones."""
    results = []
    heals_done = 0
    tracker = _CooldownTracker(
        cooldown_seconds=cooldown_seconds,
        max_heals_per_pod=max_heals_per_pod,
    )

    print(f"Monitoring namespace: {namespace}")
    print(f"Check interval: {interval}s")
    print(f"Max heals: {max_heals}")
    if llm_client:
        print("AI diagnosis: enabled")
    print()

    while heals_done < max_heals:
        print(f"[{time.strftime('%H:%M:%S')}] Checking pods...")

        problematic = get_problematic_pods(namespace)

        if not problematic:
            print("  All pods healthy")
        else:
            print(f"  Found {len(problematic)} problematic pods")

            for pod in problematic:
                print(f"    - {pod.name}: {pod.phase} (restarts: {pod.restart_count})")

                # Check cooldown and max-heals
                can_heal, skip_reason = tracker.can_heal(pod.name, pod.namespace)
                if not can_heal:
                    print(f"      Skipped: {skip_reason}")
                    continue

                # Attempt healing
                action = heal_pod(pod, force_restart=True, llm_client=llm_client)

                result = HealResult(
                    pod_name=pod.name,
                    namespace=pod.namespace,
                    original_status=pod.phase,
                    action_taken=action.action,
                    success=action.success,
                    message=action.message,
                )
                results.append(result)

                if action.success:
                    heals_done += 1
                    tracker.record_heal(pod.name, pod.namespace)
                    print(f"      -> {action.action}: {action.message}")
                else:
                    print(f"      Failed: {action.message}")

        time.sleep(interval)

    return results
