"""FlashVSR v1.1 (tiny-long pipeline) on one chunk, frame-exact. Runs inside the FlashVSR venv, not the agent's.

Differences from the upstream example script (examples/WanVSR/infer_flashvsr_v1.1_tiny_long_video.py):
- No centre crop: the upscaled frame is edge-padded up to the next multiple of 128 and the padding is cut off again,
  so the full picture survives.
- No lost frames: upstream keeps only F-4 frames of the largest 8n+1 that fits, which can drop the last frames. Here
  the input is padded with its last frame up to the smallest 8n+1 >= N+4, and exactly N frames are written.
- Low-resolution frames are prepared lazily per slice instead of holding the whole chunk upscaled in RAM.
Output: `<output-dir>/%06d.png`, N frames of round(w*scale) x round(h*scale).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time


def smallest_8n1_geq(n: int) -> int:
    return max(25, math.ceil((n - 1) / 8) * 8 + 1)  # the pipeline needs at least 3 streaming steps (F >= 25)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-dir", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--scale", type=float, default=4.0)
    ap.add_argument("--sparse-ratio", type=float, default=2.0)
    ap.add_argument("--kv-ratio", type=float, default=3.0)
    ap.add_argument("--local-range", type=int, default=11)
    ap.add_argument("--no-color-fix", action="store_true")
    args = ap.parse_args()

    import imageio
    import numpy as np
    import torch
    from PIL import Image

    # Upstream code resolves its prompt tensor and `utils` package relative to examples/WanVSR.
    wan_dir = os.path.join(args.repo_dir, "examples", "WanVSR")
    os.chdir(wan_dir)
    sys.path[:0] = [wan_dir, args.repo_dir]
    from diffsynth import FlashVSRTinyLongPipeline, ModelManager
    from utils.TCDecoder import build_tcdecoder
    from utils.utils import Causal_LQ4x_Proj

    rdr = imageio.get_reader(args.input)
    src = [np.asarray(f, dtype=np.uint8)[..., :3] for f in rdr]
    rdr.close()
    n = len(src)
    if n == 0:
        print("no frames in input", file=sys.stderr)
        return 2
    h0, w0 = src[0].shape[:2]
    sw, sh = round(w0 * args.scale), round(h0 * args.scale)
    pw, ph = math.ceil(sw / 128) * 128, math.ceil(sh / 128) * 128
    f_total = smallest_8n1_geq(n + 4)
    print(f"FLASHVSR input {w0}x{h0} x{n} -> {sw}x{sh} (padded {pw}x{ph}), F={f_total}", flush=True)

    def lq_frame(i: int) -> torch.Tensor:
        img = Image.fromarray(src[min(i, n - 1)]).resize((sw, sh), Image.BICUBIC)
        arr = np.pad(np.asarray(img), ((0, ph - sh), (0, pw - sw), (0, 0)), mode="edge")
        t = torch.from_numpy(arr).permute(2, 0, 1).float() / 127.5 - 1.0
        return t.to(torch.bfloat16)

    class LazyLQ:
        """Behaves like the 1xCxFxHxW tensor the pipeline slices along the frame axis."""

        def __init__(self) -> None:
            self._last: tuple[tuple[int, int], torch.Tensor] | None = None

        def __getitem__(self, key: tuple) -> torch.Tensor:
            fs = key[2]
            a, b, _ = fs.indices(f_total)
            if self._last and self._last[0] == (a, b):
                return self._last[1]
            frames = [lq_frame(i) for i in range(a, b)]
            t = torch.stack(frames, 1).unsqueeze(0) if frames else torch.empty(1, 3, 0, ph, pw, dtype=torch.bfloat16)
            self._last = ((a, b), t)
            return t

    mm = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    mm.load_models([os.path.join(args.model_dir, "diffusion_pytorch_model_streaming_dmd.safetensors")])
    pipe = FlashVSRTinyLongPipeline.from_model_manager(mm, device="cuda")
    proj = Causal_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1).to("cuda", dtype=torch.bfloat16)
    proj.load_state_dict(torch.load(os.path.join(args.model_dir, "LQ_proj_in.ckpt"), map_location="cpu"), strict=True)
    pipe.denoising_model().LQ_proj_in = proj.to("cuda")
    pipe.TCDecoder = build_tcdecoder(new_channels=[512, 256, 128, 128], new_latent_channels=16 + 768)
    pipe.TCDecoder.load_state_dict(torch.load(os.path.join(args.model_dir, "TCDecoder.ckpt")), strict=False)
    pipe.to("cuda")
    pipe.enable_vram_management(num_persistent_param_in_dit=None)
    pipe.init_cross_kv()
    pipe.load_models_to_device(["dit", "vae"])

    t0 = time.monotonic()
    video = pipe(
        prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1, seed=0,
        LQ_video=LazyLQ(), num_frames=f_total, height=ph, width=pw, is_full_block=False, if_buffer=True,
        topk_ratio=args.sparse_ratio * 768 * 1280 / (ph * pw), kv_ratio=args.kv_ratio,
        local_range=args.local_range, color_fix=not args.no_color_fix,
    )  # C x T x H x W in [-1, 1], T = F - 4 >= n
    if video.shape[1] < n:
        print(f"pipeline returned {video.shape[1]} frames, expected >= {n}", file=sys.stderr)
        return 3

    os.makedirs(args.output_dir, exist_ok=True)
    for t in range(n):
        frame = video[:, t, :sh, :sw].float().add(1).mul(127.5).clamp(0, 255).byte().permute(1, 2, 0).numpy()
        Image.fromarray(frame).save(os.path.join(args.output_dir, f"{t:06d}.png"), compress_level=1)
    print(f"FLASHVSR done {n} frames in {time.monotonic() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
