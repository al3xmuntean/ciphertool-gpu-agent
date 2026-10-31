"""CipherTool GPU agent: runs Tier 3 upscale chunks (SeedVR2 / FlashVSR) on a rented GPU.

The agent is stateless towards CipherTool's library: it receives a source file, upscales frame-exact chunks on request
and hands back encoded chunk files. Chunking, resume, concat, audio, verification and safe-write stay in CipherTool.
"""

VERSION = "1.0.0"
PROTOCOL = 1  # bumped on breaking API changes; CipherTool checks it via /health
