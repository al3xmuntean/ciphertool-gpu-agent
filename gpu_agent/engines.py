"""Upscale engines. Each turns a lossless chunk video into numbered PNG frames; the job then encodes them.

- seedvr2: SeedVR2 `inference_cli.py` (numz/ComfyUI-SeedVR2_VideoUpscaler), its own venv.
- flashvsr: FlashVSR v1.1 tiny-long through `flashvsr_runner.py`, its own venv (needs Block-Sparse-Attention).
- scale: FFmpeg Lanczos on CPU. Not an upscaler worth paying for: it lets the whole protocol run without a GPU.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gpu_agent.config import AgentConfig

# (cmd, cwd, env) -> raises on failure / cancel
RunFn = Callable[..., None]

FLASHVSR_REPO_ID = "JunhaoZhuang/FlashVSR-v1.1"
FLASHVSR_FILES = ["diffusion_pytorch_model_streaming_dmd.safetensors", "LQ_proj_in.ckpt", "TCDecoder.ckpt"]

SEEDVR2_MODELS = [
    "seedvr2_ema_3b_fp16.safetensors",
    "seedvr2_ema_3b_fp8_e4m3fn.safetensors",
    "seedvr2_ema_3b-Q8_0.gguf",
    "seedvr2_ema_3b-Q4_K_M.gguf",
    "seedvr2_ema_7b_fp16.safetensors",
    "seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors",
    "seedvr2_ema_7b-Q4_K_M.gguf",
    "seedvr2_ema_7b_sharp_fp16.safetensors",
]


@dataclass(frozen=True)
class EngineInfo:
    name: str
    available: bool
    reason: str
    models: list[str]
    max_frames: int | None  # longest chunk the engine handles in memory; CipherTool plans chunks below it


def list_frames(frames_dir: Path) -> list[Path]:
    """All PNGs below `frames_dir` in name order (SeedVR2 writes into a per-video subfolder)."""
    return sorted(frames_dir.rglob("*.png"), key=lambda p: p.name)


def engine_infos(cfg: AgentConfig) -> list[EngineInfo]:
    enabled = cfg.enabled_engines()
    infos = []
    seed_ok = (cfg.seedvr2_dir / "inference_cli.py").exists() and cfg.seedvr2_python.exists()
    infos.append(EngineInfo("seedvr2", seed_ok and "seedvr2" in enabled,
                            "" if seed_ok else "SeedVR2 not installed in this image", SEEDVR2_MODELS, None))
    flash_ok = (cfg.flashvsr_dir / "examples" / "WanVSR").exists() and cfg.flashvsr_python.exists()
    infos.append(EngineInfo("flashvsr", flash_ok and "flashvsr" in enabled,
                            "" if flash_ok else "FlashVSR not installed in this image (seedvr2-only image)",
                            ["v1.1-tiny-long"], 600))
    infos.append(EngineInfo("scale", "scale" in enabled, "", ["lanczos"], None))
    return infos


def run_engine(
    engine: str,
    input_video: Path,
    frames_dir: Path,
    spec: Any,
    cfg: AgentConfig,
    run: RunFn,
) -> list[Path]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    if engine == "seedvr2":
        run(seedvr2_cmd(input_video, frames_dir, spec, cfg), cwd=cfg.seedvr2_dir)
    elif engine == "flashvsr":
        ensure_flashvsr_models(cfg, run)
        run(flashvsr_cmd(input_video, frames_dir, spec, cfg), cwd=cfg.flashvsr_dir / "examples" / "WanVSR")
    elif engine == "scale":
        run([
            "ffmpeg", "-hide_banner", "-nostdin", "-y", "-v", "error", "-i", str(input_video),
            "-vf", f"scale={spec.target_width}:{spec.target_height}:flags=lanczos",
            "-fps_mode", "passthrough", "-start_number", "0", str(frames_dir / "%06d.png"),
        ])
    else:
        raise ValueError(f"unknown engine {engine}")
    return list_frames(frames_dir)


def seedvr2_cmd(input_video: Path, frames_dir: Path, spec: Any, cfg: AgentConfig) -> list[str]:
    o = spec.seedvr2
    cmd = [
        str(cfg.seedvr2_python), "inference_cli.py", str(input_video),
        "--output", str(frames_dir),
        "--output_format", "png",
        "--model_dir", str(cfg.model_dir / "SEEDVR2"),
        "--dit_model", spec.model or "seedvr2_ema_7b_fp16.safetensors",
        # SeedVR2 targets the short side; the encode step scales to the exact target size afterwards.
        "--resolution", str(min(spec.target_width, spec.target_height)),
        "--batch_size", str(o.batch_size),
        "--temporal_overlap", str(o.temporal_overlap),
        "--color_correction", o.color_correction,
        "--attention_mode", o.attention_mode,
        "--seed", str(o.seed),
    ]
    if o.blocks_to_swap > 0:
        cmd += ["--blocks_to_swap", str(o.blocks_to_swap), "--dit_offload_device", "cpu"]
    if o.vae_tiling:
        cmd += ["--vae_encode_tiled", "--vae_decode_tiled"]
    if o.compile:
        cmd += ["--compile_dit", "--compile_vae"]
    if o.chunk_size > 0:
        cmd += ["--chunk_size", str(o.chunk_size)]
    if o.input_noise_scale > 0:
        cmd += ["--input_noise_scale", f"{o.input_noise_scale:g}"]
    return cmd


def flashvsr_cmd(input_video: Path, frames_dir: Path, spec: Any, cfg: AgentConfig) -> list[str]:
    o = spec.flashvsr
    return [
        str(cfg.flashvsr_python), str(Path(__file__).with_name("flashvsr_runner.py")),
        "--repo-dir", str(cfg.flashvsr_dir),
        "--model-dir", str(cfg.model_dir / "FlashVSR-v1.1"),
        "--input", str(input_video),
        "--output-dir", str(frames_dir),
        "--scale", f"{o.scale:g}",
        "--sparse-ratio", f"{o.sparse_ratio:g}",
        "--kv-ratio", f"{o.kv_ratio:g}",
        "--local-range", str(o.local_range),
        *([] if o.color_fix else ["--no-color-fix"]),
    ]


def ensure_flashvsr_models(cfg: AgentConfig, run: RunFn) -> None:
    target = cfg.model_dir / "FlashVSR-v1.1"
    if all((target / f).exists() for f in FLASHVSR_FILES):
        return
    script = (
        "import sys; from huggingface_hub import snapshot_download; "
        "snapshot_download(sys.argv[1], local_dir=sys.argv[2], allow_patterns=sys.argv[3:])"
    )
    run([str(cfg.flashvsr_python), "-c", script, FLASHVSR_REPO_ID, str(target), *FLASHVSR_FILES])


_nvenc_cache: dict[str, bool] = {}


def nvenc_available(codec: str) -> bool:
    """NVENC encoders are listed by FFmpeg even without a usable GPU, so encode one test frame."""
    name = "hevc_nvenc" if codec == "hevc" else "h264_nvenc"
    if name not in _nvenc_cache:
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-v", "error", "-f", "lavfi", "-i", "color=s=256x256:d=0.1",
             "-frames:v", "1", "-c:v", name, "-f", "null", "-"],
            capture_output=True, text=True, check=False,
        )
        _nvenc_cache[name] = proc.returncode == 0
    return _nvenc_cache[name]


def encode_args(codec: str, cq: int, ten_bit: bool) -> list[str]:
    """High-quality constant-quality encode; NVENC when the GPU has it, x265/x264 otherwise."""
    if nvenc_available(codec):
        args = ["-c:v", "hevc_nvenc" if codec == "hevc" else "h264_nvenc",
                "-preset", "p7", "-tune", "hq", "-rc", "vbr", "-cq", str(cq), "-b:v", "0"]
        if codec == "hevc" and ten_bit:
            return [*args, "-profile:v", "main10", "-pix_fmt", "p010le"]
        return [*args, "-pix_fmt", "yuv420p"]
    if codec == "hevc":
        return ["-c:v", "libx265", "-preset", "medium", "-crf", str(cq),
                "-pix_fmt", "yuv420p10le" if ten_bit else "yuv420p"]
    return ["-c:v", "libx264", "-preset", "medium", "-crf", str(cq), "-pix_fmt", "yuv420p"]


def gpu_info() -> list[dict[str, Any]]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    gpus = []
    for line in out.strip().splitlines():
        name, total, used, util, driver = (x.strip() for x in line.split(","))
        gpus.append({"name": name, "vram_total_mb": int(total), "vram_used_mb": int(used),
                     "utilization_pct": int(util), "driver": driver})
    return gpus
