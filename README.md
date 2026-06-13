# K8s Self-Healer Agent 🏥🤖

A CLI agent that watches for CrashLoopBackOff pods, uses AI to diagnose issues, and automatically restarts/rollbacks problematic pods. Demonstrates self-healing infrastructure patterns.

## What It Does

1. **Scans** for pods in CrashLoopBackOff, ImagePullBackOff, etc.
2. **Diagnoses** issues using AI (Ollama)
3. **Heals** pods by restarting or rolling back deployments
4. **Monitors** continuously for new issues

## Quick Start

```bash
# Scan for problematic pods
python src/main.py scan

# Heal a specific pod
python src/main.py heal my-pod

# Force restart
python src/main.py heal my-pod --force

# Continuous monitoring
python src/main.py monitor

# AI diagnosis
python src/main.py diagnose my-pod
```

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   kubectl       │────▶│    Healer       │────▶│   LLM Client    │
│   (K8s API)     │     │  (detection)    │     │   (Ollama)      │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                              │                         │
                              ▼                         ▼
                        ┌─────────────────┐     ┌─────────────────┐
                        │  PodStatus      │────▶│   Diagnosis     │
                        │  (structured)   │     │   (remediation) │
                        └─────────────────┘     └─────────────────┘
```

## Example

```bash
$ python src/main.py scan

Scanning namespace: default

Found 2 problematic pods:

  api-server-abc123
    Phase: CrashLoopBackOff
    Ready: False
    Restarts: 15

  worker-def456
    Phase: ImagePullBackOff
    Ready: False
    Restarts: 0

$ python src/main.py diagnose api-server-abc123

## Root Cause
The pod is crashing due to a database connection failure. The application
tries to connect to postgres:5432 at startup but the database pod is not
ready yet.

## Immediate Fix
1. Check database status:
   kubectl get pods -l app=postgres

2. Restart the api server:
   kubectl delete pod api-server-abc123

## Prevention
- Add init container to wait for database
- Use readiness probes
- Implement retry logic in application
```

## Healing Strategies

| Strategy | Condition | Action |
|----------|-----------|--------|
| Restart | 3+ restarts | Delete pod (Deployment recreates) |
| Rollback | 10+ restarts | Rollback Deployment to previous revision |
| Scale | Multiple failures | Scale down and back up |
| Manual | Unknown issue | Report for human intervention |

## Requirements

- Python 3.11+
- kubectl configured (access to a cluster)
- Ollama running locally (or OPENAI_API_KEY)

## Installation

```bash
git clone https://github.com/AkiraKane/k8s-self-healer.git
cd k8s-self-healer
```

## Docker

```bash
docker build -t k8s-self-healer .
docker run -v ~/.kube:/root/.kube:ro k8s-self-healer python main.py scan
```

## Interview Talking Points

- **Self-Healing Infrastructure**: Automated remediation of common K8s issues
- **AI Diagnostics**: Uses LLMs to understand root causes
- **Recovery Patterns**: Restart, rollback, scale strategies
- **Continuous Monitoring**: Watches for new issues

## License

MIT
