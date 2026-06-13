"""LLM client for diagnosing pod issues."""

import json
import urllib.request
import urllib.error
import os
from dataclasses import dataclass


SYSTEM_PROMPT = """You are an expert Kubernetes administrator diagnosing pod failures.

Given a pod's status and events, explain what's wrong and suggest fixes.

Rules:
- Explain the root cause in simple terms
- Provide specific kubectl commands to diagnose further
- Suggest both immediate fixes and long-term solutions
- Use markdown formatting
- Be concise — this is for automated remediation

Output in markdown format with:
1. Root Cause
2. Immediate Fix
3. Prevention"""


def diagnose_pod(
    pod_prompt: str,
    ollama_url: str = "http://localhost:11434",
    model: str = "llama3.2",
) -> str:
    """Generate pod diagnosis."""
    user_prompt = f"""Diagnose this pod failure:

{pod_prompt}"""

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {"temperature": 0.3},
    }

    try:
        req = urllib.request.Request(
            f"{ollama_url}/api/chat",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())
            return result["message"]["content"].strip()
    except urllib.error.URLError:
        openai_key = os.environ.get("OPENAI_API_KEY")
        if openai_key:
            return _diagnose_openai(pod_prompt, openai_key)
        raise ConnectionError(
            f"Cannot connect to Ollama at {ollama_url}. "
            "Start Ollama: ollama serve"
        )


def _diagnose_openai(pod_prompt: str, api_key: str) -> str:
    """Fallback to OpenAI."""
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Diagnose this pod:\n\n{pod_prompt}"},
        ],
        "temperature": 0.3,
    }
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read())
        return result["choices"][0]["message"]["content"].strip()


def check_ollama(ollama_url: str = "http://localhost:11434") -> bool:
    """Check if Ollama is running."""
    try:
        req = urllib.request.Request(f"{ollama_url}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError):
        return False


@dataclass
class HealRecommendation:
    """Structured recommendation from the LLM."""
    action: str  # restart, rollback, scale, alert, none
    confidence: float  # 0.0 to 1.0
    reasoning: str


ACTION_SYSTEM_PROMPT = """You are a Kubernetes self-healing agent.

Given pod diagnostics, recommend a healing action.

You MUST respond with ONLY valid JSON (no markdown, no explanation) in this exact format:
{
  "action": "<restart|rollback|scale|alert|none>",
  "confidence": <0.0 to 1.0>,
  "reasoning": "<brief explanation>"
}

Rules:
- "restart": delete the pod so the controller recreates it (good for transient crashes)
- "rollback": undo the last deployment revision (good for recent bad deploys)
- "scale": scale the deployment up or down (good for resource exhaustion)
- "alert": do nothing automatically, but flag for human review
- "none": the pod does not need intervention

Prefer "restart" for low restart counts (< 5) with transient errors.
Prefer "rollback" for high restart counts (> 5) that started after a deploy.
Prefer "alert" when the root cause is ambiguous or requires domain knowledge.
Use "none" only when the pod is healthy or the issue is benign."""


def recommend_action(
    pod_name: str,
    namespace: str,
    restart_count: int,
    last_termination_reason: str,
    recent_events: list[dict],
    ollama_url: str = "http://localhost:11434",
    model: str = "llama3.2",
) -> HealRecommendation:
    """Get a structured healing recommendation from the LLM."""
    events_text = ""
    if recent_events:
        for event in recent_events[:5]:
            etype = event.get("type", "")
            reason = event.get("reason", "")
            message = event.get("message", "")
            events_text += f"  [{etype}] {reason}: {message}\n"
    else:
        events_text = "  (no recent events)\n"

    user_prompt = f"""Pod: {namespace}/{pod_name}
Restart count: {restart_count}
Last termination reason: {last_termination_reason or 'N/A'}

Recent events:
{events_text}
Recommend a healing action."""

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": ACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {"temperature": 0.2},
    }

    raw = _call_llm(payload, ollama_url)
    return _parse_recommendation(raw)


def _call_llm(payload: dict, ollama_url: str) -> str:
    """Call Ollama or OpenAI and return the raw response text."""
    try:
        req = urllib.request.Request(
            f"{ollama_url}/api/chat",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())
            return result["message"]["content"].strip()
    except urllib.error.URLError:
        openai_key = os.environ.get("OPENAI_API_KEY")
        if openai_key:
            return _call_openai(payload, openai_key)
        raise ConnectionError(
            f"Cannot connect to Ollama at {ollama_url}. "
            "Start Ollama: ollama serve"
        )


def _call_openai(payload: dict, api_key: str) -> str:
    """Call OpenAI API with a compatible payload."""
    openai_payload = {
        "model": "gpt-4o-mini",
        "messages": payload["messages"],
        "temperature": payload.get("options", {}).get("temperature", 0.3),
    }
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(openai_payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read())
        return result["choices"][0]["message"]["content"].strip()


def _parse_recommendation(raw: str) -> HealRecommendation:
    """Parse LLM JSON response into a HealRecommendation."""
    # Strip markdown code fences if present
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        # Remove opening ```json or ``` and closing ```
        lines = cleaned.split("\n")
        # Drop first and last lines (the fences)
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Return a safe default if parsing fails
        return HealRecommendation(
            action="alert",
            confidence=0.1,
            reasoning=f"Failed to parse LLM response: {raw[:200]}",
        )

    action = data.get("action", "alert")
    valid_actions = {"restart", "rollback", "scale", "alert", "none"}
    if action not in valid_actions:
        action = "alert"

    confidence = data.get("confidence", 0.5)
    try:
        confidence = float(confidence)
        confidence = max(0.0, min(1.0, confidence))
    except (TypeError, ValueError):
        confidence = 0.5

    reasoning = data.get("reasoning", "")

    return HealRecommendation(
        action=action,
        confidence=confidence,
        reasoning=reasoning,
    )
