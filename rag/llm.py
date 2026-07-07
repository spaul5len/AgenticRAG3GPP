"""Local Ollama chat and embedding helpers."""

from __future__ import annotations

from urllib.parse import urlparse

import requests

from rag import config


_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
_CHAT_TIMEOUT_SECONDS = 60
_EMBED_TIMEOUT_SECONDS = 30


def _ensure_local_ollama_url(base_url: str) -> str:
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"}:
        raise RuntimeError(f"Ollama URL must use http or https: {base_url}")
    if parsed.hostname not in _LOCAL_HOSTS:
        raise RuntimeError(
            f"Refusing to call non-local Ollama host: {base_url}. "
            "Configure Ollama on localhost for this project."
        )
    return base_url.rstrip("/")


def _post_ollama(path: str, payload: dict, timeout: int) -> dict:
    base_url = _ensure_local_ollama_url(config.OLLAMA_URL)
    url = f"{base_url}{path}"
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            "Could not connect to local Ollama. Start Ollama and confirm it is "
            f"listening at {base_url}."
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise RuntimeError(
            f"Timed out calling local Ollama at {base_url}. The model may still "
            "be loading or Ollama may be overloaded."
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Local Ollama request failed: {exc}") from exc

    try:
        return response.json()
    except ValueError as exc:
        raise RuntimeError("Local Ollama returned a non-JSON response.") from exc


def _is_ollama_404(error: RuntimeError) -> bool:
    cause = error.__cause__
    response = getattr(cause, "response", None)
    return isinstance(cause, requests.exceptions.HTTPError) and response is not None and response.status_code == 404


def _format_generate_prompt(prompt: str, system_prompt: str | None) -> str:
    if not system_prompt:
        return prompt
    return f"System:\n{system_prompt}\n\nUser:\n{prompt}"


def call_local_llm(prompt: str, system_prompt: str | None = None, model: str | None = None) -> str:
    """Call local Ollama and return assistant text.

    Uses /api/chat first. If the local Ollama server returns 404 for that endpoint,
    falls back to /api/generate for compatibility with older/local deployments.
    """

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model or config.CHAT_MODEL,
        "messages": messages,
        "stream": False,
    }

    try:
        data = _post_ollama("/api/chat", payload, timeout=_CHAT_TIMEOUT_SECONDS)
        content = data.get("message", {}).get("content")
    except RuntimeError as exc:
        if not _is_ollama_404(exc):
            raise
        generate_payload = {
            "model": model or config.CHAT_MODEL,
            "prompt": _format_generate_prompt(prompt, system_prompt),
            "stream": False,
        }
        data = _post_ollama("/api/generate", generate_payload, timeout=_CHAT_TIMEOUT_SECONDS)
        content = data.get("response")

    if not isinstance(content, str):
        raise RuntimeError("Local Ollama response did not include assistant text.")
    return content


def embed_text(text: str, model: str | None = None) -> list[float]:
    """Call the local Ollama embedding API and return one embedding vector."""

    payload = {
        "model": model or config.EMBED_MODEL,
        "prompt": text,
    }
    data = _post_ollama("/api/embeddings", payload, timeout=_EMBED_TIMEOUT_SECONDS)

    embedding = data.get("embedding")
    if not isinstance(embedding, list) or not all(
        isinstance(value, (int, float)) for value in embedding
    ):
        raise RuntimeError("Local Ollama embedding response did not include a numeric embedding.")
    return [float(value) for value in embedding]
