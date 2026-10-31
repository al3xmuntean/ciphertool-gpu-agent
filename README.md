# CipherTool GPU agent

Runs Tier 3 upscale chunks (SeedVR2, FlashVSR) on a rented GPU. CipherTool keeps the whole pipeline (chunk plan,
resume, concat, audio, verification, SafeWriter); the agent only turns frame-exact chunks of an uploaded source into
encoded chunk files. 

## Image

| Tag | Content | GPUs |
|---|---|---|
| `ghcr.io/al3xmuntean/ciphertool-gpu-agent:seedvr2` | SeedVR2 v2.5.24 (commit `4490bd1`), torch 2.8 / CUDA 12.8 | Ampere, Ada, Hopper, Blackwell |
| `ghcr.io/al3xmuntean/ciphertool-gpu-agent:full` | + FlashVSR v1.1 (commit `cf910c6`), torch 2.6 / CUDA 12.4, Block-Sparse-Attention | FlashVSR: A100/H100 class (sm_80/sm_90); RTX 40 untested upstream; not Blackwell |

The image holds only open-source code and this agent: no media, no keys. Models download on first use into
`AGENT_MODEL_DIR` (default `/workspace/models`; on RunPod mount a network volume there to keep them between pods).
Built by `.github/workflows/gpu-agent-image.yml` (on request only: a manual run, or pushing a tag `gpu-agent-seedvr2-v*` / `gpu-agent-full-v*`).

## Environment

| Variable | Default | |
|---|---|---|
| `AGENT_TOKEN` | — | required, ≥ 16 chars; every request needs `Authorization: Bearer <token>` |
| `AGENT_PORT` | `8000` | |
| `AGENT_DATA_DIR` | `/data` | sources, work files, results (container disk: gone when the pod is terminated) |
| `AGENT_MODEL_DIR` | `/workspace/models` | |
| `AGENT_IDLE_TIMEOUT_S` | `1200` | no request and no job for this long → shut down; `0` disables |
| `AGENT_SHUTDOWN_CMD` | auto | command that stops billing. Auto: `runpodctl remove pod $RUNPOD_POD_ID` on RunPod, else only process exit. Vast.ai: `vastai destroy instance $CONTAINER_ID --api-key $CONTAINER_API_KEY` (needs `vastai` installed) |
| `AGENT_ENGINES` | per image | comma list of `seedvr2`, `flashvsr`, `scale` |

## API (protocol 1)

| Route | |
|---|---|
| `GET /health` | version, protocol, GPUs, NVENC, engines (+ models, `max_frames`), free disk, busy |
| `PUT /sources/{id}?offset=N` | append one part (≤ 128 MB) at `offset`; a wrong offset → 409 with `received` |
| `GET /sources/{id}` | `received` bytes / `complete` |
| `POST /sources/{id}/complete` | `{size, sha256}`; mismatch → 422 and the upload is discarded |
| `POST /jobs` | chunk job (see `JobSpec` in `gpu_agent/jobs.py`) → 202 |
| `GET /jobs/{id}` | status, stage, progress, fps, frames_in/out, result size + SHA-256, log tail |
| `GET /jobs/{id}/result` | the encoded chunk (`.mkv`), header `X-Content-SHA256` |
| `DELETE /jobs/{id}` | cancel / delete result |
| `POST /shutdown` | stop billing now |

A job: lossless frame-exact extraction (`-ss` half a frame early + `-frames:v`, client pre-filters) → engine → PNG →
one encode at the exact target size (HEVC 10-bit NVENC CQ 18 by default; x265 without NVENC) → checks
`frames_out == frames_in`.

## Engines

- **seedvr2** — `inference_cli.py`; `--resolution` = short side of the target; `batch_size` must be 4n+1;
  `blocks_to_swap` > 0 also offloads the DiT to CPU. Models by file name, e.g. `seedvr2_ema_7b_fp16.safetensors`.
- **flashvsr** — `flashvsr_runner.py` around the upstream tiny-long pipeline, changed to keep the full frame (pad, not
  crop) and every frame (pad to 8n+1, write exactly N). Chunks up to 600 frames (output is held in RAM).
- **scale** — FFmpeg Lanczos on CPU, to test the protocol without a GPU.

## Status

Tested without a GPU: protocol, uploads, frame-exact chunks, encode, watchdog (`tests/test_gpu_agent.py`, and the
`agent` target in Docker). **Not yet run on a GPU:** the SeedVR2 and FlashVSR engines, NVENC in the container, and
`runpodctl` self-termination; the first pilot on a rented pod covers these.

## Local test

```bash
docker build --target agent --build-arg CUDA_RUNTIME=ubuntu:24.04 -t ciphertool-gpu-agent:test .
docker run --rm -p 8000:8000 -e AGENT_TOKEN=local-test-token-123456 ciphertool-gpu-agent:test
```
