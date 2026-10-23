"""Query embeddings from EmbeddingGemma 2 on another machine (the always-on Mac).

Queries must be embedded exactly as DigUp embeds them, or they land in a different part of the
vector space: the prompt "task: search result | query: ..." (code: "task: code retrieval | query:
..."), then L2-normalised. The server is anything that speaks the OpenAI embeddings API
(llama-server, LM Studio, Ollama's /v1) or Ollama's own /api/embed.

Several base URLs can be given (MagicDNS name, Tailscale IP); they're tried in order. A server that
fails is skipped for a while, so a sleeping or rebooting Mac costs one timeout, not one per search,
and search falls back to keywords meanwhile.
"""

from __future__ import annotations

import json
import logging
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


def _probe(base: str, timeout: float) -> None:
    """Fail in seconds, not after the full timeout, when the Mac is asleep or off the tailnet
    (a connection to an offline Tailscale peer hangs rather than being refused)."""
    parts = urllib.parse.urlsplit(base)
    port = parts.port or (443 if parts.scheme == "https" else 80)
    try:
        socket.create_connection((parts.hostname, port), timeout=timeout).close()
    except socket.timeout as error:
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
        self._lock = threading.Lock()

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

    def _request(self, base: str, prompts: list[str]) -> list[np.ndarray]:
        s = self.settings
        if s.embed_api == "ollama":
            url, body = f"{base}/api/embed", {"model": s.embed_model, "input": prompts, "truncate": True}
        else:
            url = f"{base}/v1/embeddings" if not base.endswith("/v1") else f"{base}/embeddings"
            body = {"model": s.embed_model, "input": prompts, "encoding_format": "float"}
        headers = {"Content-Type": "application/json"}
        if s.embed_api_key:
            headers["Authorization"] = f"Bearer {s.embed_api_key}"
        request = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=s.embed_timeout) as response:
                payload = json.loads(response.read())
        except urllib.error.HTTPError as error:
            detail = error.read()[:300].decode(errors="replace")
            if error.code in (401, 403):
                raise EmbeddingError(f"HTTP {error.code}: check DIGUP_EMBED_API_KEY ({detail})") from error
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
            if vector.ndim == 2:  # llama-server with --pooling none returns one vector per token
                raise EmbeddingError("the server returned token vectors: start llama-server with --pooling mean")
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
                results.append({"url": url, "ok": True, "ms": round((time.monotonic() - started) * 1000)})
            except Exception as error:  # report every kind of failure
                results.append({"url": url, "ok": False, "error": str(getattr(error, "reason", error))})
        return results
