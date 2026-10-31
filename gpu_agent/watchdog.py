"""Idle watchdog: a rented GPU that nobody talks to any more must stop billing on its own.

CipherTool terminates the pod itself at the end of a job, on abort and on errors; this is the safety net for the
cases it cannot (PC off, app crash, network gone).
"""

from __future__ import annotations

import logging
import os
import shlex
import shutil
import subprocess
import threading
import time
from collections.abc import Callable

from gpu_agent.config import AgentConfig

logger = logging.getLogger("gpu_agent.watchdog")


def shutdown_command(cfg: AgentConfig) -> list[str] | None:
    """The command that ends billing: AGENT_SHUTDOWN_CMD, else `runpodctl remove pod` on RunPod."""
    if cfg.shutdown_cmd:
        return shlex.split(cfg.shutdown_cmd)
    pod_id = os.environ.get("RUNPOD_POD_ID")
    if pod_id and shutil.which("runpodctl"):
        return ["runpodctl", "remove", "pod", pod_id]
    return None


class Watchdog:
    def __init__(self, cfg: AgentConfig, busy: Callable[[], bool], exit_fn: Callable[[int], None] = os._exit) -> None:
        self.cfg = cfg
        self.busy = busy
        self.exit_fn = exit_fn
        self.last_seen = time.monotonic()
        self._thread = threading.Thread(target=self._loop, name="gpu-agent-watchdog", daemon=True)

    def touch(self) -> None:
        self.last_seen = time.monotonic()

    def idle_for(self) -> float:
        return time.monotonic() - self.last_seen

    def start(self) -> None:
        if self.cfg.idle_timeout_s > 0:
            self._thread.start()

    def _loop(self) -> None:
        while True:
            time.sleep(15)
            if self.busy():
                continue  # a running job counts as activity: the client may legitimately poll rarely
            if self.idle_for() > self.cfg.idle_timeout_s:
                self.shutdown(f"idle for {self.idle_for():.0f}s")

    def shutdown(self, reason: str) -> None:
        cmd = shutdown_command(self.cfg)
        logger.warning("Shutting down (%s) with %s", reason, cmd or "process exit only")
        if cmd:
            try:
                subprocess.run(cmd, check=False, timeout=60)
            except (OSError, subprocess.SubprocessError) as exc:
                logger.error("Shutdown command failed: %s", exc)
        self.exit_fn(0)
