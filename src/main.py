#!/usr/bin/env python3
"""K8s Self-Healer Agent — detect and fix problematic pods using AI."""

import argparse
import sys
import os

from healer import (
    get_problematic_pods, get_pod_events, restart_pod,
    rollback_deployment, heal_pod, monitor_and_heal,
    PodStatus
)
from llm import diagnose_pod, check_ollama


def main():
    parser = argparse.ArgumentParser(
        description="K8s Self-Healer Agent with AI diagnostics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s scan                          # Scan for problematic pods
  %(prog)s scan -n production            # Specify namespace
  %(prog)s heal my-pod                   # Heal specific pod
  %(prog)s heal my-pod --force           # Force restart
  %(prog)s monitor                       # Continuous monitoring
  %(prog)s diagnose my-pod               # AI diagnosis
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Scan command
    scan_parser = subparsers.add_parser("scan", help="Scan for problematic pods")
    scan_parser.add_argument("-n", "--namespace", default="default",
                            help="Kubernetes namespace")

    # Heal command
    heal_parser = subparsers.add_parser("heal", help="Heal a specific pod")
    heal_parser.add_argument("pod", help="Pod name")
    heal_parser.add_argument("-n", "--namespace", default="default",
                            help="Kubernetes namespace")
    heal_parser.add_argument("--force", action="store_true",
                            help="Force restart even if not critical")

    # Monitor command
    monitor_parser = subparsers.add_parser("monitor", help="Continuous monitoring")
    monitor_parser.add_argument("-n", "--namespace", default="default",
                               help="Kubernetes namespace")
    monitor_parser.add_argument("--interval", type=int, default=30,
                               help="Check interval in seconds")
    monitor_parser.add_argument("--max-heals", type=int, default=10,
                               help="Maximum heals before stopping")

    # Diagnose command
    diagnose_parser = subparsers.add_parser("diagnose", help="AI diagnosis")
    diagnose_parser.add_argument("pod", help="Pod name")
    diagnose_parser.add_argument("-n", "--namespace", default="default",
                                help="Kubernetes namespace")

    # Common args
    parser.add_argument("--ollama-url", default="http://localhost:11434",
                        help="Ollama API URL")
    parser.add_argument("--model", default="llama3.2",
                        help="Ollama model to use")

    args = parser.parse_args()

    if not args.command:
        parser.error("Must specify a command: scan, heal, monitor, or diagnose")

    # Execute command
    if args.command == "scan":
        _handle_scan(args)
    elif args.command == "heal":
        _handle_heal(args)
    elif args.command == "monitor":
        _handle_monitor(args)
    elif args.command == "diagnose":
        _handle_diagnose(args)


def _handle_scan(args):
    """Handle scan command."""
    print(f"Scanning namespace: {args.namespace}")
    print()

    pods = get_problematic_pods(args.namespace)

    if not pods:
        print("✓ No problematic pods found")
        return

    print(f"Found {len(pods)} problematic pods:")
    print()

    for pod in pods:
        print(f"  {pod.name}")
        print(f"    Phase: {pod.phase}")
        print(f"    Ready: {pod.ready}")
        print(f"    Restarts: {pod.restart_count}")
        print()


def _handle_heal(args):
    """Handle heal command."""
    print(f"Healing pod: {args.namespace}/{args.pod}")
    print()

    # Get pod status
    pods = get_problematic_pods(args.namespace)
    pod = next((p for p in pods if p.name == args.pod), None)

    if not pod:
        print(f"Pod {args.pod} not found or not problematic")
        return

    print(f"  Phase: {pod.phase}")
    print(f"  Restarts: {pod.restart_count}")
    print()

    # Attempt healing
    action = heal_pod(pod, force_restart=args.force)

    if action.success:
        print(f"✓ {action.action}: {action.message}")
    else:
        print(f"✗ Failed: {action.message}")


def _handle_monitor(args):
    """Handle monitor command."""
    print("Starting continuous monitoring...")
    print("Press Ctrl+C to stop")
    print()

    try:
        results = monitor_and_heal(
            namespace=args.namespace,
            interval=args.interval,
            max_heals=args.max_heals,
        )

        print("\nMonitoring complete:")
        print(f"  Total heals: {len(results)}")
        print(f"  Successful: {sum(1 for r in results if r.success)}")
        print(f"  Failed: {sum(1 for r in results if not r.success)}")

    except KeyboardInterrupt:
        print("\nMonitoring stopped")


def _handle_diagnose(args):
    """Handle diagnose command."""
    # Check Ollama
    if not check_ollama(args.ollama_url):
        if not os.environ.get("OPENAI_API_KEY"):
            print("Error: Neither Ollama nor OPENAI_API_KEY available.",
                  file=sys.stderr)
            sys.exit(1)

    # Get pod status
    pods = get_problematic_pods(args.namespace)
    pod = next((p for p in pods if p.name == args.pod), None)

    if not pod:
        print(f"Pod {args.pod} not found or not problematic")
        return

    # Get events
    events = get_pod_events(args.pod, args.namespace)

    # Build prompt
    prompt = f"Pod: {args.namespace}/{args.pod}\n"
    prompt += f"Phase: {pod.phase}\n"
    prompt += f"Restarts: {pod.restart_count}\n"

    if events:
        prompt += "\nEvents:\n"
        for event in events[:5]:
            prompt += f"  [{event.get('type', '')}] {event.get('reason', '')}: {event.get('message', '')}\n"

    # Get AI diagnosis
    print(f"Diagnosing: {args.namespace}/{args.pod}")
    print()

    try:
        diagnosis = diagnose_pod(prompt, args.ollama_url, args.model)
    except ConnectionError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(diagnosis)


if __name__ == "__main__":
    main()
