"""HTTP API of the GPU agent. Every route needs `Authorization: Bearer <AGENT_TOKEN>` (the pod's port is public)."""

from __future__ import annotations

import hmac
import logging
import os
import shutil
import threading
import time

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from gpu_agent import PROTOCOL, VERSION
from gpu_agent.config import AgentConfig
from gpu_agent.engines import engine_infos, gpu_info, nvenc_available
from gpu_agent.jobs import JobManager, JobSpec
from gpu_agent.storage import SourceStore, StorageError
from gpu_agent.watchdog import Watchdog

logger = logging.getLogger("gpu_agent")

MAX_PART_BYTES = 128 * 1024 * 1024


class CompleteRequest(BaseModel):
    size: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")


def create_app(cfg: AgentConfig | None = None, start_threads: bool = True) -> FastAPI:
    cfg = cfg or AgentConfig()
    if len(cfg.token) < 16:
        raise RuntimeError("AGENT_TOKEN must be set (at least 16 characters)")
    cfg.ensure_dirs()

    sources = SourceStore(cfg.sources_dir)
    jobs = JobManager(cfg, sources)
    watchdog = Watchdog(cfg, busy=jobs.busy)
    started = time.time()

    def auth(request: Request) -> None:
        header = request.headers.get("authorization", "")
        expected = f"Bearer {cfg.token}"
        if not hmac.compare_digest(header.encode(), expected.encode()):
            raise HTTPException(status_code=401, detail="unauthorized")
        watchdog.touch()

    app = FastAPI(title="CipherTool GPU Agent", version=VERSION, dependencies=[Depends(auth)],
                  docs_url=None, redoc_url=None, openapi_url=None)
    app.state.cfg, app.state.jobs, app.state.watchdog = cfg, jobs, watchdog

    if start_threads:
        jobs.start()
        watchdog.start()

    @app.exception_handler(StorageError)
    def _storage_error(_: Request, exc: StorageError) -> JSONResponse:
        return JSONResponse(status_code=exc.status, content={"detail": str(exc), **exc.extra})

    @app.get("/health")
    def health() -> dict:
        disk = shutil.disk_usage(cfg.data_dir)
        return {
            "version": VERSION,
            "protocol": PROTOCOL,
            "uptime_s": round(time.time() - started),
            "gpus": gpu_info(),
            "nvenc": {"hevc": nvenc_available("hevc"), "h264": nvenc_available("h264")},
            "engines": [e.__dict__ for e in engine_infos(cfg)],
            "disk_free_gb": round(disk.free / 1e9, 1),
            "busy": jobs.busy(),
            "idle_timeout_s": cfg.idle_timeout_s,
        }

    # --- sources --------------------------------------------------------------------------------------------

    @app.get("/sources/{source_id}")
    def source_status(source_id: str) -> dict:
        return sources.status(source_id)

    @app.put("/sources/{source_id}")
    async def upload_part(source_id: str, request: Request, offset: int = Query(ge=0)) -> dict:
        declared = int(request.headers.get("content-length") or 0)
        if declared > MAX_PART_BYTES:
            raise HTTPException(status_code=413, detail=f"part larger than {MAX_PART_BYTES} bytes")
        body = await request.body()
        received = sources.write_part(source_id, offset, [body])
        return {"source_id": source_id, "received": received}

    @app.post("/sources/{source_id}/complete")
    def complete_source(source_id: str, req: CompleteRequest) -> dict:
        return sources.complete(source_id, req.size, req.sha256)

    @app.delete("/sources/{source_id}")
    def delete_source(source_id: str) -> dict:
        sources.delete(source_id)
        return {"deleted": source_id}

    # --- jobs -----------------------------------------------------------------------------------------------

    def _job(job_id: str):
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"job {job_id} not found")
        return job

    @app.post("/jobs", status_code=202)
    def submit(spec: JobSpec) -> dict:
        engine = next(e for e in engine_infos(cfg) if e.name == spec.engine)
        if not engine.available:
            raise HTTPException(status_code=400, detail=f"engine {spec.engine} unavailable: {engine.reason}")
        if engine.max_frames and spec.frames > engine.max_frames:
            raise HTTPException(status_code=400, detail=f"{spec.engine} handles at most {engine.max_frames} frames")
        return jobs.submit(spec).to_dict()

    @app.get("/jobs")
    def list_jobs() -> list[dict]:
        return [j.to_dict() for j in jobs.jobs.values()]

    @app.get("/jobs/{job_id}")
    def job_status(job_id: str) -> dict:
        return _job(job_id).to_dict()

    @app.get("/jobs/{job_id}/result")
    def job_result(job_id: str) -> FileResponse:
        job = _job(job_id)
        path = jobs.result_path(job)
        if job.status != "done" or not path.exists():
            raise HTTPException(status_code=409, detail=f"job {job_id} has no result (status {job.status})")
        return FileResponse(path, media_type="video/x-matroska", filename=f"chunk_{job.spec.chunk_idx:04d}.mkv",
                            headers={"X-Content-SHA256": job.result_sha256})

    @app.delete("/jobs/{job_id}")
    def delete_job(job_id: str) -> dict:
        jobs.delete(_job(job_id))
        return {"deleted": job_id}

    @app.post("/shutdown", status_code=202)
    def shutdown() -> dict:
        threading.Timer(1.0, watchdog.shutdown, args=("requested by client",)).start()
        return {"shutting_down": True}

    return app


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    uvicorn.run(create_app(), host="0.0.0.0", port=int(os.environ.get("AGENT_PORT", "8000")), log_level="info")


if __name__ == "__main__":
    main()
