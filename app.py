# app.py
from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import uuid
import os
from fastapi.responses import StreamingResponse
from fastapi.responses import RedirectResponse
import io
import csv
import json  # you already import json in benchmark_engine, but not here


from benchmark_engine import (
    BenchmarkConfig,
    BenchmarkRunner,
    BenchmarkStatus,
    DeviceType,
    identify_device,
    make_builtin_config,
    GridBenchmarkRunner,
)
import db


app = FastAPI(title="Bitaxe / NerdQaxe Benchmark Web")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# in-memory runners for active jobs
runners: Dict[str, BenchmarkRunner] = {}


class BenchmarkCreate(BaseModel):
    bitaxe_ip: str
    device_type: DeviceType
    initial_voltage: int = 1150
    initial_frequency: int = 500

    max_temp: int = 66
    max_vr_temp: int = 86
    max_allowed_voltage: int = 1400
    min_allowed_voltage: int = 1000
    max_allowed_frequency: int = 1200
    min_allowed_frequency: int = 400
    max_power: int = 40
    min_input_voltage: int = 4800
    max_input_voltage: int = 5500

    benchmark_time: int = 600
    sample_interval: int = 15
    sleep_time: int = 90
    min_samples: int = 7
    voltage_increment: int = 20
    frequency_increment: int = 25
    error_rate_warn_threshold: float = 2.0

    notes: str = ""
    profile_name: Optional[str] = None
    # NEW: "adaptive" (existing) or "grid" (full V/F sweep)
    sweep_mode: str = "adaptive"


class AutoDetectRequest(BaseModel):
    bitaxe_ip: str


class ProfileCreate(BaseModel):
    name: str
    device_type: DeviceType
    config: Dict[str, Any]


class NoteUpdate(BaseModel):
    notes: str


def builtin_profiles_for_api() -> List[Dict[str, Any]]:
    profiles = []
    for dtype in [DeviceType.GAMMA_602, DeviceType.NERDQAXE_PP, DeviceType.OTHER]:
        cfg = make_builtin_config(dtype)
        profiles.append(
            {
                "id": f"builtin:{dtype.value}",
                "name": {
                    DeviceType.GAMMA_602: "Bitaxe Gamma 602 (default)",
                    DeviceType.NERDQAXE_PP: "NerdQaxe++ (default)",
                    DeviceType.OTHER: "Generic / Other (default)",
                }[dtype],
                "deviceType": dtype.value,
                "builtin": True,
                "config": cfg.__dict__,
            }
        )
    return profiles


def on_runner_finish(runner: BenchmarkRunner, result_path: Optional[str]) -> None:
    # If the runner has results but didn't pass a path, try to write one now.
    if not result_path and getattr(runner, "results", None):
        try:
            result_path = runner._write_results_json()
        except Exception:
            # log if you have logging; don't crash the thread
            pass
    db.finish_run(
        run_id=runner.run_id,
        status=runner.status.value,
        best_hashrate=runner.best_hashrate,
        best_efficiency=runner.best_efficiency,
        result_path=result_path,
        error_reason=runner.error_reason,
    )
    runners.pop(runner.run_id, None)


@app.on_event("startup")
def on_startup():
    db.init_db()
    os.makedirs("data/results", exist_ok=True)


@app.post("/api/devices/identify")
def api_identify_device(payload: AutoDetectRequest):
    try:
        result = identify_device(payload.bitaxe_ip)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/profiles")
def api_list_profiles():
    builtin = builtin_profiles_for_api()
    custom = db.list_profiles()
    return {"profiles": builtin + custom}


@app.post("/api/profiles")
def api_create_profile(payload: ProfileCreate):
    profile_id = db.save_profile(
        name=payload.name,
        device_type=payload.device_type.value,
        config=payload.config,
    )
    return {
        "id": f"custom:{profile_id}",
        "name": payload.name,
        "deviceType": payload.device_type.value,
        "builtin": False,
        "config": payload.config,
    }


@app.delete("/api/profiles/{profile_id}")
def api_delete_profile(profile_id: str):
    if not profile_id.startswith("custom:"):
        raise HTTPException(status_code=400, detail="Cannot delete builtin profile")
    try:
        numeric_id = int(profile_id.split(":", 1)[1])
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid profile id")

    db.delete_profile(numeric_id)
    return {"status": "ok"}


@app.post("/api/benchmarks")
def api_create_benchmark(payload: BenchmarkCreate):
    normalized_ip = payload.bitaxe_ip.strip()

    # Prevent duplicate active runs against same target
    for runner in runners.values():
        if (
            runner.bitaxe_ip_raw == normalized_ip
            and runner.status in (BenchmarkStatus.RUNNING, BenchmarkStatus.PENDING)
        ):
            raise HTTPException(
                status_code=400,
                detail=f"A benchmark is already running for device {normalized_ip}.",
            )

    run_id = str(uuid.uuid4())

    # Build BenchmarkConfig from payload
    cfg_dict = payload.dict()
    cfg_dict["device_type"] = payload.device_type
    field_names = set(BenchmarkConfig.__dataclass_fields__.keys())
    subset = {k: v for k, v in cfg_dict.items() if k in field_names}
    cfg = BenchmarkConfig(**subset)

    # Save run metadata in DB
    db.create_run(
        run_id=run_id,
        bitaxe_ip=normalized_ip,
        device_type=payload.device_type.value,
        status=BenchmarkStatus.RUNNING.value,
        profile_name=payload.profile_name,
        notes=payload.notes,
        config=cfg_dict,
    )
    
    # NEW: pick runner type based on sweep_mode
    RunnerClass = GridBenchmarkRunner if cfg.sweep_mode == "grid" else BenchmarkRunner

    runner = RunnerClass(
        run_id=run_id,
        bitaxe_ip=normalized_ip,
        config=cfg,
        on_finish=on_runner_finish,
    )
    runners[run_id] = runner
    runner.start()
    return {"runId": run_id}


@app.get("/api/benchmarks")
def api_list_benchmarks():
    rows = db.list_runs()
    for r in rows:
        rid = r["id"]
        if rid in runners:
            runner = runners[rid]
            live = runner.to_dict(include_results=True)
            r["status"] = live["status"]
            r["best_hashrate"] = live["bestHashrate"]
            r["best_efficiency"] = live["bestEfficiency"]
            r["eta_seconds"] = live.get("etaSeconds")
            r["progress"] = live.get("progress", 0.0)
            r["status_detail"] = live.get("statusDetail")
            r["recent_results"] = live.get("recentResults", [])
            r["results"] = live.get("results", [])
            r["error_reason"] = live.get("errorReason")   # ← NEW v0.1.1
            r["config"] = live.get("config")
    return {"runs": rows}


@app.get("/api/benchmarks/{run_id}")
def api_get_benchmark(run_id: str, include_results: bool = Query(False)):
    # Prefer live runner data if still running
    if run_id in runners:
        runner = runners[run_id]
        data = runner.to_dict(include_results=include_results)
        meta = db.get_run(run_id)

        # Make response look more like finished runs (snake_case fields)
        data["id"] = run_id
        data["bitaxe_ip"] = getattr(runner, "bitaxe_ip_raw", None)
        data["device_type"] = runner.config.device_type.value

        if meta:
            data["notes"] = meta.get("notes")
            data["profile_name"] = meta.get("profile_name")
            data["dbStatus"] = meta.get("status")
            # Keep original DB fields if you want them:
            # e.g. started_at, finished_at, etc.
            if meta.get("started_at"):
                data["started_at"] = meta["started_at"]
            if meta.get("finished_at"):
                data["finished_at"] = meta["finished_at"]

        return data


    meta = db.get_run(run_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Run not found")

    data: Dict[str, Any] = dict(meta)
    if meta.get("config_json"):
        try:
            # config_json is stored as a JSON string; parse it so the frontend
            # can access sweep_mode, etc., as an object.
            data["config"] = json.loads(meta["config_json"])
        except Exception:
            # Fallback: at least return the raw string
            data["config"] = meta["config_json"]

    if include_results and meta.get("result_path"):
        try:
            with open(meta["result_path"], "r") as f:
                import json

                data["resultsFile"] = json.load(f)
        except Exception:
            data["resultsFile"] = None

    return data


@app.post("/api/benchmarks/{run_id}/cancel")
def api_cancel_benchmark(run_id: str):
    runner = runners.get(run_id)
    if runner:
        runner.cancel()
        runner.error_reason = "Cancelled via API"
        return {"status": "cancel_requested"}

    # No in-memory runner: treat as stale run
    meta = db.get_run(run_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Run not found")

    status = (meta.get("status") or "").lower()
    if status not in ("running", "pending"):
        # Already completed/failed/cancelled; nothing to do
        return {"status": f"not_running (current status={status})"}

    # Mark it cancelled in the DB so UI can delete it
    db.update_run_status(
        run_id,
        BenchmarkStatus.CANCELLED.value,
        error_reason="Cancelled with no active runner (likely container restart)",
    )

    return {"status": "marked_cancelled"}



@app.delete("/api/benchmarks/{run_id}")
def api_delete_benchmark(run_id: str):
    """
    Delete a completed / failed / cancelled run and its results file.
    Active runs must be cancelled first.
    """
    # Don't allow deleting active in-memory runs
    runner = runners.get(run_id)
    if runner and runner.status in (BenchmarkStatus.RUNNING, BenchmarkStatus.PENDING):
        raise HTTPException(
            status_code=400,
            detail="Cannot delete an active benchmark. Cancel it first.",
        )

    meta = db.get_run(run_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Run not found")

    status = (meta.get("status") or "").lower()
    if status in ("running", "pending") and runner is not None:
        # DB thinks it's active even if runner isn't; be conservative
        raise HTTPException(
            status_code=400,
            detail="Cannot delete a benchmark marked as active. Mark it cancelled/failed first.",
        )

    # Remove result file if present
    result_path = meta.get("result_path")
    if result_path and os.path.exists(result_path):
        try:
            os.remove(result_path)
        except OSError:
            # best-effort; don't hard-fail on filesystem
            pass

    # Delete DB row
    conn = db._get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM benchmark_runs WHERE id = ?;", (run_id,))
    conn.commit()
    conn.close()

    return {"status": "deleted", "runId": run_id}

@app.get("/api/benchmarks/{run_id}/csv")
def api_get_benchmark_csv(run_id: str):
    meta = db.get_run(run_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Run not found")

    result_path = meta.get("result_path")
    if not result_path or not os.path.exists(result_path):
        raise HTTPException(status_code=404, detail="Result file not found for this run")

    with open(result_path, "r") as f:
        data = json.load(f)

    results = data.get("results") or []
    if not results:
        raise HTTPException(status_code=400, detail="No results available for this run")

    # Define CSV columns – matches what the engine writes (plus averagePower now)
    fieldnames = [
        "coreVoltage",
        "frequency",
        "averageHashRate",
        "averageTemperature",
        "averageVRTemp",
        "averagePower",
        "efficiencyJTH",
        "fanSpeed",
        "errorPercentage",
        "rejectRate",
        "sharesAccepted",
        "sharesRejected",
        "sharesTotal",
    ]

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for row in results:
        writer.writerow({k: row.get(k) for k in fieldnames})

    output.seek(0)
    filename = f"benchmark_{run_id}.csv"
    return StreamingResponse(
        output,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )

@app.api_route("/api/benchmarks/{run_id}/notes", methods=["POST", "PUT"])
def api_update_notes(run_id: str, payload: NoteUpdate):
    meta = db.get_run(run_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Run not found")
    conn = db._get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE benchmark_runs SET notes = ? WHERE id = ?;
        """,
        (payload.notes, run_id),
    )
    conn.commit()
    conn.close()
    return {"status": "ok"}

@app.get("/", include_in_schema=False)
async def root(request: Request):
    # This respects root_path and any proxy prefixes
    url = request.url_for("static", path="index.html")
    return RedirectResponse(url=url)