
# dashboard_api.py
from __future__ import annotations

from fastapi import APIRouter, HTTPException, UploadFile, File, Body, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone
import time
import os
import json
import ipaddress
import hashlib
import mimetypes
import sqlite3
import re
import socket
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

import db

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])

ASSET_ROOT = os.path.join("data", "dashboard_assets")
BG_DIR = os.path.join(ASSET_ROOT, "backgrounds")
SND_DIR = os.path.join(ASSET_ROOT, "sounds")

DEFAULT_SETTINGS: Dict[str, Any] = {
    "refresh_interval_ms": 5000,
    "request_timeout_s": 1.2,
    "block_odds_timescale": "day",  # hour|day|month|year
    "theme": "dark",  # "dark" or "light" (front-end also keeps bb_theme localStorage)
    "clean_mode": False,
    "card_transparency_pct": 8,
    "hashrate_unit": "GH",
    "hashrate_decimals": 2,
    "rejected_share_red_threshold_pct": 1.0,
    "max_columns": 0,  # 0 = auto
    "compact_cards": True,
    "enable_scan": True,
    "scan_default_cidr": "192.168.1.0/24",
    "animations": {
        "enabled": True,
        "coin_drop": True,
        "shake_on_share": True,
        "sound_on_share": False,
        "sound_volume": 0.35,
        "max_coins": 35,
    },
    "thresholds": {
        "chip_temp": {
            "warn": 60.0,
            "danger": 70.0,
            "warn_color": "#f59e0b",
            "danger_color": "#ef4444",
        },
        "vrm_temp": {
            "warn": 70.0,
            "danger": 85.0,
            "warn_color": "#f59e0b",
            "danger_color": "#ef4444",
        },
        "hashrate": {
            "warn_pct_of_10m": 70.0,
            "warn_color": "#f59e0b",
        },
        "offline": {
            "grace_s": 15,
        },
    },
    "assets": {
        "active_background_id": None,
        "active_sound_id": None,
    },
}


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_dirs() -> None:
    os.makedirs(BG_DIR, exist_ok=True)
    os.makedirs(SND_DIR, exist_ok=True)


def _ensure_tables() -> None:
    _ensure_dirs()
    conn = db._get_conn()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS dashboard_devices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            ip TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL,
            sort_order INTEGER NOT NULL DEFAULT 0,
            poll_type TEXT NOT NULL DEFAULT 'http',
            last_seen TEXT,
            last_poll TEXT,
            online INTEGER NOT NULL DEFAULT 0,
            last_error TEXT,
            last_info_json TEXT
        );
        """
    )


    # ---- schema migrations ----
    # Add poll_type (http / avalon_cgminer / auto) for protocol-aware polling.
    cur.execute("PRAGMA table_info(dashboard_devices);")
    cols = {row[1] for row in cur.fetchall()}  # (cid, name, type, notnull, dflt_value, pk)
    if "poll_type" not in cols:
        cur.execute("ALTER TABLE dashboard_devices ADD COLUMN poll_type TEXT NOT NULL DEFAULT 'http';")
    # normalize any empty values
    cur.execute("UPDATE dashboard_devices SET poll_type='http' WHERE poll_type IS NULL OR TRIM(poll_type)='';")

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS dashboard_settings (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            settings_json TEXT NOT NULL
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS dashboard_assets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            kind TEXT NOT NULL,              -- 'background' or 'sound'
            filename TEXT NOT NULL,
            orig_name TEXT,
            mime TEXT,
            size_bytes INTEGER,
            created_at TEXT NOT NULL,
            active INTEGER NOT NULL DEFAULT 0
        );
        """
    )

    conn.commit()
    conn.close()


# Ensure tables exist at import time (keeps app.py changes minimal).
_ensure_tables()


def _deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Merge b into a (copy), recursively for dicts."""
    out = json.loads(json.dumps(a))
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _get_settings() -> Dict[str, Any]:
    conn = db._get_conn()
    cur = conn.cursor()
    cur.execute("SELECT settings_json FROM dashboard_settings WHERE id = 1;")
    row = cur.fetchone()
    conn.close()
    if not row:
        return json.loads(json.dumps(DEFAULT_SETTINGS))
    try:
        stored = json.loads(row["settings_json"])
    except Exception:
        stored = {}
    return _deep_merge(DEFAULT_SETTINGS, stored)


def _save_settings(settings: Dict[str, Any]) -> None:
    conn = db._get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO dashboard_settings (id, settings_json)
        VALUES (1, ?)
        ON CONFLICT(id) DO UPDATE SET settings_json=excluded.settings_json;
        """,
        (json.dumps(settings),),
    )
    conn.commit()
    conn.close()


class DeviceCreate(BaseModel):
    ip: str = Field(..., description="IPv4/IPv6 address")
    name: Optional[str] = None


class SettingsUpdate(BaseModel):
    settings: Dict[str, Any]


class ReorderPayload(BaseModel):
    device_ids: List[int]


def _validate_ip(ip: str) -> str:
    try:
        return str(ipaddress.ip_address(ip.strip()))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid IP: {ip}") from e


def _list_devices() -> List[Dict[str, Any]]:
    conn = db._get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT * FROM dashboard_devices
        ORDER BY sort_order ASC, id ASC;
        """
    )
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _get_latest_benchmark_for_ip(ip: str) -> Optional[Dict[str, Any]]:
    conn = db._get_conn()
    cur = conn.cursor()
    # Prefer most recently finished completed run; fall back to most recent run.
    cur.execute(
        """
        SELECT id, status, started_at, finished_at
        FROM benchmark_runs
        WHERE bitaxe_ip = ?
        ORDER BY COALESCE(finished_at, started_at) DESC
        LIMIT 1;
        """,
        (ip,),
    )
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def _write_device_poll(
    device_id: int,
    online: bool,
    info: Optional[Dict[str, Any]],
    error: Optional[str],
    poll_type: Optional[str] = None,
) -> None:
    conn = db._get_conn()
    cur = conn.cursor()
    now = _utcnow_iso()
    last_seen = now if online else None
    cur.execute(
        """
        UPDATE dashboard_devices
        SET online = ?,
            last_poll = ?,
            last_seen = COALESCE(?, last_seen),
            last_error = ?,
            last_info_json = COALESCE(?, last_info_json),
            poll_type = COALESCE(?, poll_type)
        WHERE id = ?;
        """,
        (
            1 if online else 0,
            now,
            last_seen,
            error,
            json.dumps(info) if info is not None else None,
            poll_type,
            device_id,
        ),
    )
    conn.commit()
    conn.close()


def _cgminer_query(ip: str, cmd: str, timeout_s: float) -> str:
    # Avalon Q runs a cgminer-compatible TCP API on port 4028.
    # It expects the raw command string (no newline).

                                                                                   
                                                                              
       
                   
                                                     
              
    with socket.create_connection((ip, 4028), timeout=timeout_s) as s:
        s.settimeout(timeout_s)
        s.sendall(cmd.encode("utf-8", errors="ignore"))
        buf = b""
                                      
                       
                
        while True:
                
            chunk = s.recv(4096)
            if not chunk:
                break
            buf += chunk
            # cgminer responses end in a pipe delimiter
            if b"|" in chunk:
                break
            if len(buf) > 250_000:
                break
                         
        return buf.decode("utf-8", errors="replace")


def _parse_cgminer_sections(resp: str) -> List[Dict[str, str]]:
    # Split "STATUS=...|SUMMARY,...|" into a list of dicts per section
                                                             
                                                           
       
    out: List[Dict[str, str]] = []
    for sec in (resp or "").split("|"):
        sec = sec.strip()
        if not sec:
            continue
        d: Dict[str, str] = {}
        for part in sec.split(","):
                        
                                      
                    
                            
             
                                
                            
            if "=" not in part:
                continue
            k, v = part.split("=", 1)
                        
                 
                                          
                                                    
                         
                  


                                                          


                                                    
       
                                                                         
                                                                          
       
                            

                                 
            d[k.strip()] = v.strip()
        if d:
                       

                           
                
                                    
                             
                        

                                                 
                          
                  
                                
                     
                           
                    
            out.append(d)
                                 
                        
                                             
                                     
                                  
                             
                                    

                
            
                         
                         
                
                               
                             
                        

                                                 
                                                       
                            
                 
                    
                             
    return out


def _pick_first(sections: List[Dict[str, str]], key: str) -> Optional[Dict[str, str]]:
    for d in sections:
        if key in d:
                   
                    
                                    
            return d
    return None


def _probe_avalon_q(ip: str, timeout_s: float) -> Tuple[bool, Optional[Dict[str, str]], Optional[str]]:
    try:
        v_secs = _parse_cgminer_sections(_cgminer_query(ip, "version", timeout_s))
        ver = _pick_first(v_secs, "PROD") or _pick_first(v_secs, "MODEL") or {}
        if not ver:
            return False, None, "No cgminer version response"
        return True, ver, None
    except Exception as e:
                      
                       
                     
        return False, None, str(e)


def _poll_avalon_q(ip: str, timeout_s: float) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:

                                                            
                            
       
    try:
        v_secs = _parse_cgminer_sections(_cgminer_query(ip, "version", timeout_s))
        s_secs = _parse_cgminer_sections(_cgminer_query(ip, "summary", timeout_s))
        e_secs = _parse_cgminer_sections(_cgminer_query(ip, "estats", timeout_s))

        ver = _pick_first(v_secs, "PROD") or {}
        summ = _pick_first(s_secs, "Elapsed") or {}
        stats = _pick_first(e_secs, "STATS") or {}

        # hashrate: cgminer returns MHS (mega-hash/s). Convert to GH/s for dashboard parity.
        def mhs_to_gh(v: Any) -> Optional[float]:
            try:
                x = float(v)
                return x / 1000.0
            except Exception:
                return None

        hr_now = mhs_to_gh(summ.get("MHS 5s"))
        hr_1m = mhs_to_gh(summ.get("MHS 1m"))
        hr_5m = mhs_to_gh(summ.get("MHS 5m"))
        hr_15m = mhs_to_gh(summ.get("MHS 15m"))
        hr_avg = mhs_to_gh(summ.get("MHS av"))
                                 
                                                                                             
                                 

        # temps: Avalon estats exposes several: ITemp, HBITemp, HBOTemp, TAvg, TMax.
        def num(v: Any) -> Optional[float]:
            try:
                return float(v)
            except Exception:
                return None

        chip_temp = num(stats.get("TAvg") or stats.get("HBOTemp") or stats.get("HBITemp"))
        vrm_temp = num(stats.get("HBITemp") or stats.get("ITemp"))
                          
                                                                 
                                                                                           
                                  
                                        

        # fan: prefer FanR (percent), fall back to rpm average
                                                 
                                                 
        fan_pct = None
        fr = stats.get("FanR")
        if isinstance(fr, str) and fr.endswith("%"):
            fan_pct = num(fr[:-1])
        if fan_pct is None:
            fan_pct = num(fr)

        fan_rpms = [num(stats.get(k)) for k in ("Fan1", "Fan2", "Fan3", "Fan4")]
                                                                                  
                                            
                                         
                                            
                                                                    
                                                                                   
        fan_rpms_f = [x for x in fan_rpms if x is not None]
        fan_rpm_avg = (sum(fan_rpms_f) / len(fan_rpms_f)) if fan_rpms_f else None
                                                                                     

        # shares / best diff
                         
                         
        acc = num(summ.get("Accepted"))
        rej = num(summ.get("Rejected"))
        best = num(summ.get("Best Share"))
                       
                       
                       
                                             
                                                                          
                                                                                   
                                             
                                        
                                       
                                             
                                    

        # identity / firmware
        prod = ver.get("PROD") or "Avalon"
        model = ver.get("MODEL") or ""
        device_model = (f"{prod} {model}").strip()
        lver = ver.get("LVERSION") or ver.get("CGVERSION") or ""
        mac = ver.get("MAC") or None
                                                                             

        # elasped
        up = None
        try:
            up = int(float(summ.get("Elapsed"))) if summ.get("Elapsed") is not None else None
        except Exception:
            up = None

        # Build a dashboard-shaped info blob
                                   
                                      
                            
                                       
                   
         

                        
                                                     
                                                     
                                                      
                                                             

        info: Dict[str, Any] = {
            "deviceModel": device_model or "Avalon",
            "hostname": f"{device_model}" if device_model else "Avalon",
            "version": str(lver) if lver else None,
                                                                                      
            "macAddr": mac,
                         
                             
                                                                       
                           

            "uptimeSeconds": up,
            "hashRate": hr_now,
            "hashRate_1m": hr_1m,
            "hashRate_10m": hr_5m,   # closest cgminer provides
            "hashRate_1h": hr_avg,   # best long-ish signal available
            "temp": chip_temp,
            "vrTemp": vrm_temp,

            "fanspeed": fan_pct,
            "fanrpm": fan_rpm_avg,

            "sharesAccepted": int(acc) if acc is not None else None,
            "sharesRejected": int(rej) if rej is not None else None,
            "bestDiff": best,

            "foundBlocks": int(float(summ.get("Found Blocks"))) if summ.get("Found Blocks") else None,
                         
                              
                                   
                                 
                             
                                    
                                  
                                  
         

            # raw for detail view / debugging
            "_avalon": {
                "version": ver,
                "summary": summ,
                "estats": stats,
            },
                     
        }

        return True, info, None
    except Exception as e:
        return False, None, str(e)


def _fetch_system_info(
    ip: str,
    timeout_s: float,
    poll_type: str = "auto",
) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str], str]:
    pt = (poll_type or "auto").strip().lower()

    # Explicit Avalon polling
    if pt in ("avalon", "avalon_q", "cgminer", "avalon_cgminer"):
        ok, info, err = _poll_avalon_q(ip, timeout_s)
        return ok, info, err, "avalon_cgminer"

    # Explicit HTTP polling
    if pt in ("http", "bitaxe", "nerdqaxe"):
        url = f"http://{ip}/api/system/info"
        try:
            r = requests.get(url, timeout=timeout_s)
            r.raise_for_status()
            data = r.json()
            if not isinstance(data, dict) or "deviceModel" not in data:
                # still accept but mark as suspicious
                return True, data if isinstance(data, dict) else {"raw": data}, None, "http"
            return True, data, None, "http"
        except Exception as e:
            return False, None, str(e), "http"

    # Auto-detect:
    # Try Avalon first *quickly* to avoid waiting for HTTP timeouts before probing cgminer.
    quick = max(0.15, min(0.45, timeout_s))
    ok_probe, _ver, err_a = _probe_avalon_q(ip, quick)
    if ok_probe:
        ok_full, info_a, err_full = _poll_avalon_q(ip, timeout_s)
        return ok_full, info_a, err_full, "avalon_cgminer"

    # then try HTTP with the full timeout
    url = f"http://{ip}/api/system/info"
                                  
    try:
        r = requests.get(url, timeout=timeout_s)
        r.raise_for_status()
        data = r.json()
                           
        if not isinstance(data, dict) or "deviceModel" not in data:
                                                 
            return True, data if isinstance(data, dict) else {"raw": data}, None, "http"
        return True, data, None, "http"
                                          
    except Exception as e:
        # prefer HTTP error, but include Avalon probe info if it looked meaningful
        extra = f" (avalon probe: {err_a})" if err_a else ""
                                                 
          
                               
        return False, None, str(e) + extra, "auto"


@router.get("/settings")
def api_get_settings():
    return {"settings": _get_settings()}


@router.post("/settings")
def api_update_settings(payload: SettingsUpdate):
    current = _get_settings()
    merged = _deep_merge(current, payload.settings or {})
    _save_settings(merged)
    return {"status": "ok", "settings": merged}


# ---- Network difficulty helper (for block-odds UI) ----
_DIFFICULTY_CACHE: Dict[str, Any] = {"difficulty": None, "source": None, "as_of": None, "fetched_at": 0.0}

def _fetch_difficulty_from_mempool(api_base: str = "https://mempool.space", timeout_s: float = 2.5) -> float:
    """
    Uses a public mempool/esplora-compatible REST API:
      - GET /api/blocks/tip/hash -> tip block hash
      - GET /api/block/:hash -> JSON includes "difficulty"
    """
    tip_hash = requests.get(f"{api_base}/api/blocks/tip/hash", timeout=timeout_s).text.strip()
    if not tip_hash:
        raise RuntimeError("Empty tip hash")
    blk = requests.get(f"{api_base}/api/block/{tip_hash}", timeout=timeout_s).json()
    diff = blk.get("difficulty")
    if diff is None:
        raise RuntimeError("No difficulty in block payload")
    return float(diff)

def _get_network_difficulty() -> Dict[str, Any]:
    # 1) Prefer device-provided value (Bitaxe Gamma exposes 'networkDifficulty')
    best_diff: Optional[float] = None
    best_as_of: Optional[str] = None
    try:
        rows = db.get_devices()
    except Exception:
        rows = []
    for d in rows:
        info_json = d.get("last_info_json")
        if not info_json:
            continue
        try:
            info = json.loads(info_json)
        except Exception:
            continue
        diff = info.get("networkDifficulty") or info.get("difficulty")
        try:
            diff_f = float(diff)
        except Exception:
            continue
        if not (diff_f > 0):
            continue
        as_of = d.get("last_poll") or d.get("last_seen") or None
        # keep the newest timestamp we can parse
        if best_as_of is None:
            best_diff, best_as_of = diff_f, as_of
            continue
        try:
            cur_ts = datetime.fromisoformat(str(as_of).replace("Z", "+00:00")).timestamp() if as_of else 0.0
        except Exception:
            cur_ts = 0.0
        try:
            best_ts = datetime.fromisoformat(str(best_as_of).replace("Z", "+00:00")).timestamp() if best_as_of else 0.0
        except Exception:
            best_ts = 0.0
        if cur_ts >= best_ts:
            best_diff, best_as_of = diff_f, as_of

    if best_diff is not None:
        return {"difficulty": best_diff, "source": "device", "as_of": best_as_of}

    # 2) Cache (avoid hammering public APIs)
    now = time.time()
    if _DIFFICULTY_CACHE.get("difficulty") is not None and (now - float(_DIFFICULTY_CACHE.get("fetched_at") or 0)) < 43200:
        return {
            "difficulty": _DIFFICULTY_CACHE["difficulty"],
            "source": _DIFFICULTY_CACHE.get("source") or "cache",
            "as_of": _DIFFICULTY_CACHE.get("as_of"),
            "cached": True,
        }

    # 3) Public fallback
    diff = _fetch_difficulty_from_mempool("https://mempool.space")
    payload = {"difficulty": diff, "source": "mempool.space", "as_of": _utcnow_iso()}
    _DIFFICULTY_CACHE.update({**payload, "fetched_at": now})
    return payload


@router.get("/network/difficulty")
def api_get_network_difficulty():
    """
    Returns current Bitcoin network difficulty.
    Prefers miners that expose it; otherwise falls back to a public mempool/esplora endpoint.
    """
    try:
        return _get_network_difficulty()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch difficulty: {e}")



@router.get("/devices")
def api_list_devices():
    devices = _list_devices()
    # light normalization for client convenience
    out = []
    for d in devices:
        info = None
        try:
            if d.get("last_info_json"):
                info = json.loads(d["last_info_json"])
        except Exception:
            info = None
        out.append(
            {
                "id": d["id"],
                "ip": d["ip"],
                "name": d.get("name"),
                "sort_order": d.get("sort_order", 0),
                "online": bool(d.get("online", 0)),
                "last_seen": d.get("last_seen"),
                "last_poll": d.get("last_poll"),
                "last_error": d.get("last_error"),
                "poll_type": d.get("poll_type") or "http",
                "last_info": info,
                "latest_benchmark": _get_latest_benchmark_for_ip(d["ip"]),
            }
        )
    return {"devices": out}


@router.post("/devices")
def api_add_device(payload: DeviceCreate):
    ip = _validate_ip(payload.ip)
    conn = db._get_conn()
    cur = conn.cursor()
    now = _utcnow_iso()
    # put at end
    cur.execute("SELECT COALESCE(MAX(sort_order), 0) + 1 AS next_order FROM dashboard_devices;")
    next_order = int(cur.fetchone()["next_order"])
    try:
        cur.execute(
            """
            INSERT INTO dashboard_devices (name, ip, created_at, sort_order, poll_type)
            VALUES (?, ?, ?, ?, ?);
            """,
            (payload.name, ip, now, next_order, "auto"),
        )
        conn.commit()
    except sqlite3.IntegrityError:  # type: ignore[name-defined]
        conn.close()
        raise HTTPException(status_code=409, detail="Device already exists")

    poll_type_final = "auto"

    # Quick protocol hint: if cgminer TCP/4028 answers, it's very likely an Avalon Q.
    # This avoids the first dashboard refresh doing an HTTP timeout before discovering it.
    try:
        ok_a, _ver, _err = _probe_avalon_q(ip, 0.35)
        if ok_a:
            cur.execute("UPDATE dashboard_devices SET poll_type=? WHERE ip=?;", ("avalon_cgminer", ip))
            conn.commit()
            poll_type_final = "avalon_cgminer"
    except Exception:
        pass

    device_id = cur.lastrowid
    conn.close()
    return {
        "status": "ok",
        "device": {"id": device_id, "ip": ip, "name": payload.name, "sort_order": next_order, "poll_type": poll_type_final},
    }




@router.delete("/devices/{device_id}")
def api_delete_device(device_id: int):
    conn = db._get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM dashboard_devices WHERE id = ?;", (device_id,))
    if cur.rowcount <= 0:
        conn.close()
        raise HTTPException(status_code=404, detail="Device not found")
    conn.commit()
    conn.close()
    return {"status": "deleted"}


@router.post("/devices/reorder")
def api_reorder_devices(payload: ReorderPayload):
    ids = payload.device_ids or []
    conn = db._get_conn()
    cur = conn.cursor()

    # Validate IDs exist
    if ids:
        cur.execute(
            f"SELECT id FROM dashboard_devices WHERE id IN ({','.join(['?']*len(ids))});",
            ids,
        )
        existing = {int(r["id"]) for r in cur.fetchall()}
        missing = [i for i in ids if i not in existing]
        if missing:
            conn.close()
            raise HTTPException(status_code=400, detail=f"Unknown device ids: {missing}")

    for idx, did in enumerate(ids):
        cur.execute("UPDATE dashboard_devices SET sort_order = ? WHERE id = ?;", (idx, did))

    conn.commit()
    conn.close()
    return {"status": "ok"}


@router.get("/status")
def api_poll_status(
    timeout_s: Optional[float] = Query(None, ge=0.2, le=10.0),
    parallel: int = Query(32, ge=1, le=128),
):
    """
    Poll each saved device and return aggregated status (HTTP BitAxe-style API or Avalon cgminer TCP/4028).
    Also updates the DB with last_poll/last_seen/last_info_json.
    """
    settings = _get_settings()
    timeout = float(timeout_s if timeout_s is not None else settings.get("request_timeout_s", 1.2))

    devices = _list_devices()
    results: List[Dict[str, Any]] = []
    now = _utcnow_iso()

    def work(d: Dict[str, Any]) -> Dict[str, Any]:
        pt = (d.get("poll_type") or "auto")
        ok, info, err, detected = _fetch_system_info(d["ip"], timeout, poll_type=pt)
        poll_update = detected if ok and detected in ("http", "avalon_cgminer") else None
        _write_device_poll(d["id"], ok, info, None if ok else err, poll_type=poll_update)
        latest = _get_latest_benchmark_for_ip(d["ip"])
        return {
            "id": d["id"],
            "ip": d["ip"],
            "name": d.get("name"),
            "poll_type": (poll_update or d.get("poll_type") or "http"),
            "online": ok,
            "info": info,
            "error": err,
            "last_poll": now,
            "latest_benchmark": latest,
        }

    with ThreadPoolExecutor(max_workers=parallel) as ex:
        futures = [ex.submit(work, d) for d in devices]
        for f in as_completed(futures):
            results.append(f.result())

    # preserve sort order from DB
    order = {d["id"]: (d.get("sort_order", 0), d["id"]) for d in devices}
    results.sort(key=lambda r: order.get(r["id"], (10_000, r["id"])))

    return {"now": now, "devices": results}


class ScanPayload(BaseModel):
    cidr: str = Field(..., description="CIDR, e.g. 192.168.10.0/24")
    timeout_s: float = Field(0.8, ge=0.2, le=10.0)
    parallel: int = Field(64, ge=1, le=128)
    limit: int = Field(512, ge=1, le=2048)


@router.post("/scan")
def api_scan(payload: ScanPayload):
    """
    Scan a CIDR for devices that respond to /api/system/info.
    Returns a list of found systems (does NOT auto-add).
    """
    try:
        net = ipaddress.ip_network(payload.cidr, strict=False)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CIDR: {payload.cidr}") from e

    hosts = [str(h) for h in net.hosts()]
    if len(hosts) > payload.limit:
        raise HTTPException(
            status_code=400,
            detail=f"Refusing to scan {len(hosts)} hosts (limit {payload.limit}). Use a smaller CIDR or raise limit.",
        )

    found: List[Dict[str, Any]] = []
    timeout = float(payload.timeout_s)

    def probe(ip: str) -> Optional[Dict[str, Any]]:
        ok, info, err, detected = _fetch_system_info(ip, timeout, poll_type='auto')
        if not ok:
            return None
        # attach a few convenient fields
        if isinstance(info, dict):
            hostname = info.get("hostname") or info.get("host") or None
            model = info.get("deviceModel") or info.get("ASICModel") or None
        else:
            hostname, model = None, None
        return {"ip": ip, "hostname": hostname, "model": model, "info": info}

    with ThreadPoolExecutor(max_workers=int(payload.parallel)) as ex:
        futures = [ex.submit(probe, ip) for ip in hosts]
        for f in as_completed(futures):
            item = f.result()
            if item:
                found.append(item)

    found.sort(key=lambda x: x.get("ip", ""))
    return {"cidr": payload.cidr, "found": found, "count": len(found)}


def _safe_filename(original: str, content: bytes) -> str:
    h = hashlib.sha256(content).hexdigest()[:16]
    base = os.path.basename(original or "file")
    base = base.replace(" ", "_")
    root, ext = os.path.splitext(base)
    ext = (ext or "").lower()[:12]
    if ext and not re.match(r"^\.[a-z0-9]+$", ext):
        ext = ""
    return f"{root[:32]}_{h}{ext}"


@router.get("/assets")
def api_list_assets(kind: str = Query("background", pattern="^(background|sound)$")):
    conn = db._get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT * FROM dashboard_assets
        WHERE kind = ?
        ORDER BY active DESC, created_at DESC, id DESC;
        """,
        (kind,),
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()

    active_id = None
    for r in rows:
        if r.get("active"):
            active_id = r["id"]
            break
    return {"kind": kind, "assets": rows, "active_id": active_id}


@router.post("/assets/upload")
async def api_upload_asset(
    kind: str = Query("background", pattern="^(background|sound)$"),
    file: UploadFile = File(...),
):
    _ensure_dirs()
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty upload")

    # Basic size limits (keep it sane inside a docker volume)
    max_bytes = 50 * 1024 * 1024 if kind == "background" else 10 * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(status_code=400, detail=f"File too large (limit {max_bytes} bytes)")

    mime = file.content_type or mimetypes.guess_type(file.filename or "")[0] or "application/octet-stream"
    out_dir = BG_DIR if kind == "background" else SND_DIR
    os.makedirs(out_dir, exist_ok=True)

    # Deduplicate by sha (same filename may differ; we hash content)
    sha = hashlib.sha256(content).hexdigest()
    filename = f"{sha[:16]}_{os.path.basename(file.filename or 'asset')}".replace(" ", "_")[:90]
    path = os.path.join(out_dir, filename)

    # If file already exists, don't rewrite
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(content)

    conn = db._get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO dashboard_assets (kind, filename, orig_name, mime, size_bytes, created_at, active)
        VALUES (?, ?, ?, ?, ?, ?, 0);
        """,
        (kind, filename, file.filename, mime, len(content), _utcnow_iso()),
    )
    asset_id = int(cur.lastrowid)
    conn.commit()
    conn.close()
    return {"status": "ok", "asset": {"id": asset_id, "kind": kind, "filename": filename, "mime": mime}}


@router.post("/assets/{asset_id}/activate")
def api_activate_asset(asset_id: int, kind: str = Query("background", pattern="^(background|sound)$")):
    conn = db._get_conn()
    cur = conn.cursor()
    # verify exists
    cur.execute("SELECT id FROM dashboard_assets WHERE id = ? AND kind = ?;", (asset_id, kind))
    row = cur.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Asset not found")

    # deactivate others
    cur.execute("UPDATE dashboard_assets SET active = 0 WHERE kind = ?;", (kind,))
    cur.execute("UPDATE dashboard_assets SET active = 1 WHERE id = ?;", (asset_id,))
    conn.commit()
    conn.close()

    # Also store in settings for convenience
    s = _get_settings()
    if kind == "background":
        s["assets"]["active_background_id"] = asset_id
    else:
        s["assets"]["active_sound_id"] = asset_id
    _save_settings(s)

    return {"status": "ok", "active_id": asset_id}


@router.delete("/assets/{asset_id}")
def api_delete_asset(asset_id: int, kind: str = Query("background", pattern="^(background|sound)$")):
    conn = db._get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM dashboard_assets WHERE id = ? AND kind = ?;", (asset_id, kind))
    row = cur.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Asset not found")

    row = dict(row)

    filename = row["filename"]
    active = bool(row.get("active"))
    cur.execute("DELETE FROM dashboard_assets WHERE id = ?;", (asset_id,))
    conn.commit()
    conn.close()

    out_dir = BG_DIR if kind == "background" else SND_DIR
    path = os.path.join(out_dir, filename)
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass

    if active:
        # clear from settings too
        s = _get_settings()
        if kind == "background":
            s["assets"]["active_background_id"] = None
        else:
            s["assets"]["active_sound_id"] = None
        _save_settings(s)

    return {"status": "deleted"}


@router.get("/assets/{asset_id}/file")
def api_asset_file(asset_id: int):
    conn = db._get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM dashboard_assets WHERE id = ?;", (asset_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Asset not found")
        
    row = dict(row)

    kind = row["kind"]
    filename = row["filename"]
    out_dir = BG_DIR if kind == "background" else SND_DIR
    path = os.path.join(out_dir, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Asset file missing")
    media_type = row.get("mime") or None
    return FileResponse(path, media_type=media_type, filename=row.get("orig_name") or filename)
