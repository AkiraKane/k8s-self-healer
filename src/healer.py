"""Kubernetes self-healing agent for detecting and fixing problematic pods."""

import subprocess
import json
import time
from dataclasses import dataclass, field
from typing import Optional


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
           "-o", "jsonpath='{.metadata.ownerReferences[0].name}'"]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        return result.stdout.strip().strip("'")

    return None


def heal_pod(pod: PodStatus, force_restart: bool = False) -> HealAction:
    """Attempt to heal a problematic pod."""
    # Get deployment name
    deployment = get_deployment_for_pod(pod.name, pod.namespace)

    # If high restart count, try rollback
    if pod.restart_count > 10 and deployment:
        return rollback_deployment(deployment, pod.namespace)

    # Otherwise restart
    if force_restart or pod.restart_count > 3:
        return restart_pod(pod.name, pod.namespace)

    return HealAction(
        pod_name=pod.name,
        namespace=pod.namespace,
        action="none",
        reason="Pod not in critical state yet"
    )


def monitor_and_heal(namespace: str = "default",
                     interval: int = 30,
                     max_heals: int = 10) -> list[HealResult]:
    """Monitor pods and heal problematic ones."""
    results = []
    heals_done = 0

    print(f"Monitoring namespace: {namespace}")
    print(f"Check interval: {interval}s")
    print(f"Max heals: {max_heals}")
    print()

    while heals_done < max_heals:
        print(f"[{time.strftime('%H:%M:%S')}] Checking pods...")

        problematic = get_problematic_pods(namespace)

        if not problematic:
            print("  ✓ All pods healthy")
        else:
            print(f"  ⚠ Found {len(problematic)} problematic pods")

            for pod in problematic:
                print(f"    - {pod.name}: {pod.phase} (restarts: {pod.restart_count})")

                # Attempt healing
                action = heal_pod(pod, force_restart=True)

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
                    print(f"      → {action.action}: {action.message}")
                else:
                    print(f"      ✗ Failed: {action.message}")

        time.sleep(interval)

    return results
