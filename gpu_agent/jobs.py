"""Chunk jobs: one GPU, one worker thread, FIFO queue.

A job = extract the chunk's frames from an uploaded source (frame-exact, with CipherTool's pre-filters) into a
lossless video -> engine -> PNG frames -> one high-quality encode at the exact target size -> result file + SHA-256.
"""

from __future__ import annotations

import logging
import os
import queue
import re
import shutil
import subprocess
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from gpu_agent.config import AgentConfig
from gpu_agent.engines import encode_args, list_frames, run_engine
from gpu_agent.storage import SourceStore, check_id, sha256_file

logger = logging.getLogger("gpu_agent.jobs")

# Only these filters may come from the client (they reach FFmpeg as one -vf argument, never a shell).
_PRE_FILTER_RE = re.compile(r"^(bwdif|yadif|hqdn3d|nlmeans|deblock|setsar|null)(=[A-Za-z0-9_.:=-]*)?$")
_FPS_RE = re.compile(r"^\d+(\.\d+)?(/\d+)?$")


class SeedVR2Options(BaseModel):
    batch_size: int = Field(21, ge=1, le=129)
    temporal_overlap: int = Field(4, ge=0, le=16)
    color_correction: Literal["lab", "wavelet", "wavelet_adaptive", "hsv", "adain", "none"] = "lab"
    attention_mode: Literal["sdpa", "flash_attn_2", "flash_attn_3", "sageattn_2", "sageattn_3"] = "sdpa"
    blocks_to_swap: int = Field(0, ge=0, le=36)
    vae_tiling: bool = False
    compile: bool = False
    chunk_size: int = Field(0, ge=0)
    input_noise_scale: float = Field(0.0, ge=0.0, le=1.0)
    seed: int = 42

    @field_validator("batch_size")
    @classmethod
    def _four_n_plus_one(cls, v: int) -> int:
        if v % 4 != 1:
            raise ValueError("batch_size must be 4n+1 (1, 5, 9, 13, 17, 21, ...)")
        return v


class FlashVSROptions(BaseModel):
    scale: float = Field(4.0, ge=1.5, le=4.0)
    sparse_ratio: float = Field(2.0, ge=1.0, le=3.0)
    kv_ratio: float = Field(3.0, ge=1.0, le=4.0)
    local_range: Literal[9, 11] = 11
    color_fix: bool = True


class EncodeSpec(BaseModel):
    codec: Literal["hevc", "h264"] = "hevc"
    cq: int = Field(18, ge=8, le=35)
    ten_bit: bool = True


class JobSpec(BaseModel):
    source_id: str
    chunk_idx: int = Field(ge=0)
    start_frame: int = Field(ge=0)
    frames: int = Field(ge=1, le=100_000)
    seek_s: float = Field(ge=0.0)
    fps: str
    target_width: int = Field(ge=16, le=8192)
    target_height: int = Field(ge=16, le=8192)
    pre_filters: list[str] = Field(default_factory=list)
    engine: Literal["seedvr2", "flashvsr", "scale"]
    model: str = ""
    seedvr2: SeedVR2Options = Field(default_factory=SeedVR2Options)
    flashvsr: FlashVSROptions = Field(default_factory=FlashVSROptions)
    encode: EncodeSpec = Field(default_factory=EncodeSpec)

    @field_validator("source_id")
    @classmethod
    def _id(cls, v: str) -> str:
        return check_id(v)

    @field_validator("fps")
    @classmethod
    def _fps(cls, v: str) -> str:
        if not _FPS_RE.match(v):
            raise ValueError("fps must look like 25, 23.976 or 24000/1001")
        return v

    @field_validator("target_width", "target_height")
    @classmethod
    def _even(cls, v: int) -> int:
        if v % 2:
            raise ValueError("target dimensions must be even")
        return v

    @field_validator("pre_filters")
    @classmethod
    def _filters(cls, v: list[str]) -> list[str]:
        for f in v:
            if not _PRE_FILTER_RE.match(f):
                raise ValueError(f"pre-filter not allowed: {f}")
        return v

    @field_validator("model")
    @classmethod
    def _model(cls, v: str) -> str:
        if v and not re.match(r"^[A-Za-z0-9_.-]{1,128}$", v):
            raise ValueError("invalid model name")
        return v


class JobCancelled(Exception):
    pass


class JobFailed(Exception):
    pass


class Job:
    def __init__(self, spec: JobSpec) -> None:
        self.id = uuid.uuid4().hex[:16]
        self.spec = spec
        self.status: Literal["queued", "running", "done", "failed", "cancelled"] = "queued"
        self.stage = "queued"
        self.progress = 0.0
        self.fps = 0.0
        self.error = ""
        self.frames_in = 0
        self.frames_out = 0
        self.result_size = 0
        self.result_sha256 = ""
        self.created_at = time.time()
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.cancel = threading.Event()
        self.log: deque[str] = deque(maxlen=40)

    def to_dict(self) -> dict:
        return {
            "job_id": self.id,
            "chunk_idx": self.spec.chunk_idx,
            "engine": self.spec.engine,
            "model": self.spec.model,
            "status": self.status,
            "stage": self.stage,
            "progress": round(self.progress, 1),
            "fps": round(self.fps, 3),
            "error": self.error,
            "frames_in": self.frames_in,
            "frames_out": self.frames_out,
            "result_size": self.result_size,
            "result_sha256": self.result_sha256,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "log_tail": list(self.log)[-10:],
        }


class JobManager:
    def __init__(self, cfg: AgentConfig, sources: SourceStore) -> None:
        self.cfg = cfg
        self.sources = sources
        self.jobs: dict[str, Job] = {}
        self._queue: queue.Queue[Job] = queue.Queue()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._worker, name="gpu-agent-worker", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def submit(self, spec: JobSpec) -> Job:
        self.sources.path(spec.source_id)  # 404 early if the source is not complete
        job = Job(spec)
        with self._lock:
            self.jobs[job.id] = job
        self._queue.put(job)
        return job

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def busy(self) -> bool:
        return any(j.status in ("queued", "running") for j in self.jobs.values())

    def result_path(self, job: Job) -> Path:
        return self.cfg.results_dir / f"{job.id}.mkv"

    def delete(self, job: Job) -> None:
        job.cancel.set()
        self.result_path(job).unlink(missing_ok=True)
        with self._lock:
            self.jobs.pop(job.id, None)

    # --- worker ------------------------------------------------------------------------------------------------

    def _worker(self) -> None:
        while True:
            job = self._queue.get()
            if job.cancel.is_set():
                job.status = "cancelled"
                continue
            job.status, job.started_at = "running", time.time()
            try:
                self._process(job)
                job.status, job.stage, job.progress = "done", "done", 100.0
            except JobCancelled:
                job.status, job.stage = "cancelled", "cancelled"
            except Exception as exc:  # report every failure to the client instead of killing the worker
                logger.exception("job %s failed", job.id)
                job.status, job.error = "failed", str(exc)[-2000:]
            finally:
                job.finished_at = time.time()
                shutil.rmtree(self.cfg.work_dir / job.id, ignore_errors=True)

    def _run(self, job: Job, cmd: list[str], cwd: Path | None = None, tick=None) -> None:
        """Run a step, killing it on cancel; stdout/stderr tail goes to the job log and error."""
        logger.info("job %s: %s", job.id, " ".join(cmd))
        # Lower priority for the heavy work, so the HTTP API keeps answering status polls during CPU encodes.
        nice = {"preexec_fn": lambda: os.nice(10)} if hasattr(os, "nice") else {}
        proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                errors="replace", **nice)
        lines: deque[str] = deque(maxlen=60)

        def _reader() -> None:
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.rstrip()
                if line:
                    lines.append(line)
                    job.log.append(line[-300:])

        reader = threading.Thread(target=_reader, daemon=True)
        reader.start()
        while proc.poll() is None:
            if job.cancel.is_set():
                proc.kill()
                proc.wait()
                raise JobCancelled()
            if tick:
                tick()
            time.sleep(1.0)
        reader.join(timeout=5)
        if proc.returncode != 0:
            raise JobFailed(f"{Path(cmd[0]).name} exited with {proc.returncode}:\n" + "\n".join(list(lines)[-25:]))

    def _process(self, job: Job) -> None:
        spec, cfg = job.spec, self.cfg
        job_dir = cfg.work_dir / job.id
        frames_dir, enc_dir = job_dir / "frames", job_dir / "enc"
        enc_dir.mkdir(parents=True, exist_ok=True)
        lossless = job_dir / "input.mkv"

        # 1. Frame-exact, lossless chunk extraction with the client's pre-filters.
        job.stage = "extract"
        cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-y", "-v", "error"]
        if spec.seek_s > 0:
            cmd += ["-ss", f"{spec.seek_s:.6f}"]
        cmd += ["-i", str(self.sources.path(spec.source_id)), "-frames:v", str(spec.frames),
                "-vf", ",".join(spec.pre_filters) or "null", "-an", "-sn", "-dn",
                "-c:v", "libx264", "-qp", "0", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(lossless)]
        self._run(job, cmd)
        job.frames_in = count_frames(lossless)
        if job.frames_in == 0:
            raise JobFailed("no frames extracted (seek past the end of the source?)")
        job.progress = 5.0

        # 2. Upscale. Progress = PNG frames written so far.
        job.stage = "upscale"
        t0 = time.monotonic()

        def tick() -> None:
            done = len(list_frames(frames_dir)) if frames_dir.exists() else 0
            job.progress = 5.0 + 85.0 * min(1.0, done / job.frames_in)
            elapsed = time.monotonic() - t0
            job.fps = done / elapsed if elapsed > 0 else 0.0

        frames = run_engine(spec.engine, lossless, frames_dir, spec, cfg,
                            lambda c, cwd=None: self._run(job, c, cwd=cwd, tick=tick))
        upscale_s = time.monotonic() - t0
        if len(frames) < job.frames_in:
            raise JobFailed(f"engine wrote {len(frames)} frames, expected {job.frames_in}")
        # Some engines pad the batch; extra trailing frames are not part of the chunk.
        for i, f in enumerate(frames[: job.frames_in]):
            f.rename(enc_dir / f"{i:06d}.png")
        job.fps = job.frames_in / upscale_s if upscale_s > 0 else 0.0

        # 3. Encode once, at the exact target size and final quality.
        job.stage, job.progress = "encode", 90.0
        result = self.result_path(job)
        partial = result.with_suffix(".partial.mkv")
        self._run(job, [
            "ffmpeg", "-hide_banner", "-nostdin", "-y", "-v", "error",
            "-framerate", spec.fps, "-start_number", "0", "-i", str(enc_dir / "%06d.png"),
            "-vf", f"scale={spec.target_width}:{spec.target_height}:flags=lanczos,setsar=1",
            *encode_args(spec.encode.codec, spec.encode.cq, spec.encode.ten_bit),
            "-an", str(partial),
        ])
        job.frames_out = count_frames(partial)
        if job.frames_out != job.frames_in:
            raise JobFailed(f"encoded {job.frames_out} frames, expected {job.frames_in}")
        partial.rename(result)
        job.result_size = result.stat().st_size
        job.result_sha256 = sha256_file(result)


def count_frames(path: Path) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=False,
    ).stdout.strip()
    return int(out) if out.isdigit() else 0
