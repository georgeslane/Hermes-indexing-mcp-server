"""Settings: environment variables, then an optional KEY=VALUE file, then defaults.

Hermes passes `env:` from its config.yaml to the server, and the CLI (`hermes-digup doctor`)
reads the same file, so the shell and the agent always see the same settings.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_CONFIG_FILE = Path("~/.config/hermes-digup/config.env").expanduser()
DEFAULT_INDEX_DIR = Path("~/.local/share/hermes-digup/index").expanduser()


def read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return values
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        values[key] = value
    return values


@dataclass
class Settings:
    index_dir: Path = DEFAULT_INDEX_DIR
    # One or more base URLs, tried in order (e.g. MagicDNS name, then the Tailscale IP).
    embed_urls: list[str] = field(default_factory=list)
    embed_api: str = "openai"  # "openai" (llama-server, LM Studio, Ollama /v1) or "ollama" (/api/embed)
    embed_model: str = "embeddinggemma-2"
    embed_api_key: str = ""
    embed_timeout: float = 15.0
    stale_hours: float = 36.0
    log_level: str = "INFO"
    config_file: Path | None = None

    @classmethod
    def load(cls, environ: dict[str, str] | None = None) -> Settings:
        env = dict(os.environ if environ is None else environ)
        config_file = Path(env.get("DIGUP_CONFIG", str(DEFAULT_CONFIG_FILE))).expanduser()
        merged = {**read_env_file(config_file), **{k: v for k, v in env.items() if v != ""}}

        def get(key: str, default: str = "") -> str:
            return merged.get(key, default).strip()

        urls = [u.strip().rstrip("/") for u in get("DIGUP_EMBED_URL").split(",") if u.strip()]
        api = get("DIGUP_EMBED_API", "openai").lower()
        if api not in ("openai", "ollama"):
            raise ValueError(f"DIGUP_EMBED_API must be 'openai' or 'ollama', not {api!r}")
        return cls(
            index_dir=Path(get("DIGUP_INDEX_DIR", str(DEFAULT_INDEX_DIR))).expanduser(),
            embed_urls=urls,
            embed_api=api,
            embed_model=get("DIGUP_EMBED_MODEL", "embeddinggemma-2"),
            embed_api_key=get("DIGUP_EMBED_API_KEY"),
            embed_timeout=float(get("DIGUP_EMBED_TIMEOUT", "15")),
            stale_hours=float(get("DIGUP_STALE_HOURS", "36")),
            log_level=get("DIGUP_LOG_LEVEL", "INFO").upper(),
            config_file=config_file if config_file.exists() else None,
        )
