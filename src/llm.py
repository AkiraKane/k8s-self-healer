"""LLM client for diagnosing pod issues."""

import json
import urllib.request
import urllib.error
import os


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
