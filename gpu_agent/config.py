"""Agent configuration from environment variables (set by the pod template / CipherTool when creating the pod)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name) or default)


@dataclass
class AgentConfig:
    token: str = field(default_factory=lambda: os.environ.get("AGENT_TOKEN", ""))
    # Media lives on the container disk (wiped when the pod is terminated); models may live on a network volume.
    data_dir: Path = field(default_factory=lambda: _env_path("AGENT_DATA_DIR", "/data"))
    model_dir: Path = field(default_factory=lambda: _env_path("AGENT_MODEL_DIR", "/models"))
    seedvr2_dir: Path = field(default_factory=lambda: _env_path("SEEDVR2_DIR", "/opt/seedvr2"))
    seedvr2_python: Path = field(default_factory=lambda: _env_path("SEEDVR2_PYTHON", "/opt/venv-seedvr2/bin/python"))
    flashvsr_dir: Path = field(default_factory=lambda: _env_path("FLASHVSR_DIR", "/opt/flashvsr"))
    flashvsr_python: Path = field(default_factory=lambda: _env_path("FLASHVSR_PYTHON", "/opt/venv-flashvsr/bin/python"))
    # Watchdog: with no authenticated request for this long and no job running, the agent shuts the machine down.
    idle_timeout_s: float = field(default_factory=lambda: float(os.environ.get("AGENT_IDLE_TIMEOUT_S", "1200")))
    # Command that stops billing for this machine; empty = auto (runpodctl on RunPod, otherwise only exit).
    shutdown_cmd: str = field(default_factory=lambda: os.environ.get("AGENT_SHUTDOWN_CMD", ""))
    # Engines that may run (comma list); "scale" is the CPU/ffmpeg test engine.
    engines: str = field(default_factory=lambda: os.environ.get("AGENT_ENGINES", "seedvr2,flashvsr,scale"))

    @property
    def sources_dir(self) -> Path:
        return self.data_dir / "sources"

    @property
    def work_dir(self) -> Path:
        return self.data_dir / "work"

    @property
    def results_dir(self) -> Path:
        return self.data_dir / "results"

    def enabled_engines(self) -> set[str]:
        return {e.strip() for e in self.engines.split(",") if e.strip()}

    def ensure_dirs(self) -> None:
        for d in (self.sources_dir, self.work_dir, self.results_dir, self.model_dir):
            d.mkdir(parents=True, exist_ok=True)
