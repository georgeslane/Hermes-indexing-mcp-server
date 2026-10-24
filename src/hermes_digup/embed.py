"""Query embeddings from EmbeddingGemma 2 on another machine (the always-on Mac).

Queries must be embedded exactly as DigUp embeds them, or they land in a different part of the
vector space: the prompt "task: search result | query: ..." (code: "task: code retrieval | query:
..."), then L2-normalised. The server is anything that speaks the OpenAI embeddings API
(oMLX, llama-server, LM Studio, Ollama's /v1) or Ollama's own /api/embed. None of them may add a
prompt of its own: oMLX passes the text through as it is.

Several base URLs can be given (MagicDNS name, Tailscale IP); they're tried in order. A server that
fails is skipped for a while, so a sleeping or rebooting Mac costs one timeout, not one per search,
and search falls back to keywords meanwhile.
"""

from __future__ import annotations

import json
import logging
import re
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import OrderedDict

import numpy as np

from .config import Settings
from .index import DIMENSION

log = logging.getLogger(__name__)


def query_prompt(text: str, code: bool = False) -> str:
    return f"task: {'code retrieval' if code else 'search result'} | query: {text}"


def document_prompt(title: str | None, text: str) -> str:
    return f"title: {title or 'none'} | text: {text}"


class EmbeddingError(RuntimeError):
    pass


class ModelNotFound(EmbeddingError):
    pass


# EmbeddingGemma 2 under the names servers give it: "embeddinggemma-2", "embeddinggemma-2-bf16",
# "google/embeddinggemma-2", "EmbeddingGemma2-8bit"... but not the first EmbeddingGemma ("embeddinggemma-300m").
_GEMMA2 = re.compile(r"embedding-?gemma-?2(?![0-9])", re.IGNORECASE)
# Re-quantized copies drift from DigUp's vectors; full precision is preferred when a server has both.
_QUANTIZED = re.compile(r"(?<![0-9])[2-8]-?bit|\bq[2-8]|oq|dwq|awq|gptq|nvfp4|mxfp4", re.IGNORECASE)


def pick_model(names: list[str]) -> str | None:
    """The EmbeddingGemma 2 among a server's models, full precision first."""
    candidates = sorted({n for n in names if _GEMMA2.search(n)}, key=lambda n: (bool(_QUANTIZED.search(n)), len(n), n))
    return candidates[0] if candidates else None


def _probe(base: str, timeout: float) -> None:
    """Fail in seconds, not after the full timeout, when the Mac is asleep or off the tailnet
    (a connection to an offline Tailscale peer hangs rather than being refused)."""
    parts = urllib.parse.urlsplit(base)
    port = parts.port or (443 if parts.scheme == "https" else 80)
    try:
        socket.create_connection((parts.hostname, port), timeout=timeout).close()
    except TimeoutError as error:
        raise ConnectionError(f"no connection within {timeout:g} s (asleep, or off the tailnet?)") from error


class Embedder:
    def __init__(self, settings: Settings, cache_size: int = 512, cooldown: float = 30.0,
                 connect_timeout: float = 3.0):
        self.settings = settings
        self.connect_timeout = connect_timeout
        self.cooldown = cooldown
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_size = cache_size
        self._down_until: dict[str, float] = {}
        self._last_error: str | None = None
        # The model's name on each server, when it isn't DIGUP_EMBED_MODEL (found after a 404).
        self._models: dict[str, str] = {}
        self._lock = threading.Lock()

    def model_for(self, base: str) -> str:
        return self._models.get(base, self.settings.embed_model)

    @property
    def configured(self) -> bool:
        return bool(self.settings.embed_urls)

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def embed_query(self, text: str, code: bool = False) -> np.ndarray:
        return self.embed(query_prompt(text, code))

    def embed(self, prompt: str) -> np.ndarray:
        with self._lock:
            cached = self._cache.get(prompt)
            if cached is not None:
                self._cache.move_to_end(prompt)
                return cached
        vector = self._embed_uncached([prompt])[0]
        with self._lock:
            self._cache[prompt] = vector
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return vector

    def embed_many(self, prompts: list[str]) -> list[np.ndarray]:
        return self._embed_uncached(prompts)

    def _embed_uncached(self, prompts: list[str]) -> list[np.ndarray]:
        if not self.settings.embed_urls:
            raise EmbeddingError("no embedding server configured (set DIGUP_EMBED_URL)")
        now = time.monotonic()
        urls = self.settings.embed_urls
        # Servers in cooldown go last, not away: if they're all down, still try them all.
        ordered = sorted(urls, key=lambda u: self._down_until.get(u, 0) > now)
        errors = []
        for url in ordered:
            for attempt in range(2):
                started = time.monotonic()
                try:
                    _probe(url, self.connect_timeout)
                    vectors = self._request(url, prompts)
                    self._down_until.pop(url, None)
                    self._last_error = None
                    return vectors
                except EmbeddingError as error:
                    # A wrong model or bad answer won't fix itself on a retry.
                    errors.append(f"{url}: {error}")
                    break
                except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as error:
                    # Retry once only what failed fast (a reset, a 503 while the model loads), never a timeout.
                    if attempt == 0 and time.monotonic() - started < 1.0:
                        time.sleep(0.4)
                        continue
                    errors.append(f"{url}: {getattr(error, 'reason', error)}")
                    break
            self._down_until[url] = time.monotonic() + self.cooldown
        self._last_error = "; ".join(errors)
        raise EmbeddingError(self._last_error)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.settings.embed_api_key:
            headers["Authorization"] = f"Bearer {self.settings.embed_api_key}"
        return headers

    def _request(self, base: str, prompts: list[str]) -> list[np.ndarray]:
        try:
            return self._post(base, prompts, self.model_for(base))
        except ModelNotFound as missing:
            # The server names the model differently (oMLX uses the folder name, "embeddinggemma-2-bf16"):
            # use its EmbeddingGemma 2 if it has one.
            names = self._server_models(base)
            found = pick_model(names)
            if found is None or found == self.model_for(base):
                listed = ", ".join(names) if names else "none listed"
                raise EmbeddingError(f"{missing} It has no EmbeddingGemma 2 model (it has: {listed}).") from None
            log.warning("%s has no model %r; using its %r (set DIGUP_EMBED_MODEL=%s to say so)",
                        base, self.model_for(base), found, found)
            vectors = self._post(base, prompts, found)
            self._models[base] = found
            return vectors

    def _server_models(self, base: str) -> list[str]:
        """The model names a server lists (OpenAI's /v1/models, or Ollama's /api/tags)."""
        if self.settings.embed_api == "ollama":
            url, key, field_name = f"{base}/api/tags", "models", "name"
        else:
            url = f"{base}/v1/models" if not base.endswith("/v1") else f"{base}/models"
            key, field_name = "data", "id"
        request = urllib.request.Request(url, headers=self._headers())
        try:
            with urllib.request.urlopen(request, timeout=self.settings.embed_timeout) as response:
                payload = json.loads(response.read())
            return [str(m[field_name]) for m in payload.get(key, []) if isinstance(m, dict) and field_name in m]
        except (urllib.error.URLError, OSError, ValueError, AttributeError):
            return []

    def _post(self, base: str, prompts: list[str], model: str) -> list[np.ndarray]:
        s = self.settings
        if s.embed_api == "ollama":
            url, body = f"{base}/api/embed", {"model": model, "input": prompts, "truncate": True}
        else:
            url = f"{base}/v1/embeddings" if not base.endswith("/v1") else f"{base}/embeddings"
            body = {"model": model, "input": prompts, "encoding_format": "float"}
        request = urllib.request.Request(url, data=json.dumps(body).encode(), headers=self._headers(), method="POST")
        try:
            with urllib.request.urlopen(request, timeout=s.embed_timeout) as response:
                payload = json.loads(response.read())
        except urllib.error.HTTPError as error:
            detail = error.read()[:300].decode(errors="replace")
            if error.code in (401, 403):
                raise EmbeddingError(f"HTTP {error.code}: check DIGUP_EMBED_API_KEY ({detail})") from error
            if error.code == 404 or (error.code == 400 and "not found" in detail.lower()):
                raise ModelNotFound(f"The server has no model {model!r}.") from error
            if error.code >= 500 or error.code == 429:
                raise ConnectionError(f"HTTP {error.code}: {detail}") from error
            raise EmbeddingError(f"HTTP {error.code}: {detail}") from error
        except ValueError as error:
            raise EmbeddingError(f"not JSON: {error}") from error

        try:
            if s.embed_api == "ollama":
                raw = payload["embeddings"]
            else:
                raw = [item["embedding"] for item in sorted(payload["data"], key=lambda d: d.get("index", 0))]
        except (KeyError, TypeError) as error:
            raise EmbeddingError(f"unexpected answer: {str(payload)[:200]}") from error
        if len(raw) != len(prompts):
            raise EmbeddingError(f"asked for {len(prompts)} vectors, got {len(raw)}")
        vectors = []
        for values in raw:
            vector = np.asarray(values, dtype=np.float32)
            if vector.ndim == 2:  # a server that doesn't pool returns one vector per token
                raise EmbeddingError("the server returned one vector per token instead of one per text: "
                                     "it isn't serving the model as an embedding model")
            if vector.shape != (DIMENSION,):
                raise EmbeddingError(
                    f"the server's vectors have {vector.size} dimensions, DigUp's have {DIMENSION}: "
                    "it isn't running EmbeddingGemma 2")
            norm = float(np.linalg.norm(vector))
            if not np.isfinite(norm) or norm == 0:
                raise EmbeddingError("the server returned an empty vector")
            vectors.append(vector / norm)
        return vectors

    def health(self) -> list[dict]:
        """Each server's answer to a test query, for status and doctor."""
        results = []
        for url in self.settings.embed_urls:
            started = time.monotonic()
            try:
                _probe(url, self.connect_timeout)
                self._request(url, [query_prompt("health check")])
                results.append({"url": url, "ok": True, "ms": round((time.monotonic() - started) * 1000),
                                "model": self.model_for(url)})
            except Exception as error:  # report every kind of failure
                results.append({"url": url, "ok": False, "error": str(getattr(error, "reason", error))})
        return results
