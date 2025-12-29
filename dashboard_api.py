
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
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

import db

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])

# Resolve all on-disk paths relative to this file (not the process CWD). This
# avoids accidental "multiple DB/files" situations in Docker/Portainer.
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ASSET_ROOT = os.path.join(_BASE_DIR, "data", "dashboard_assets")
BG_DIR = os.path.join(ASSET_ROOT, "backgrounds")
SND_DIR = os.path.join(ASSET_ROOT, "sounds")

BUILTIN_ASSET_ROOT = os.path.join(_BASE_DIR, "builtin_assets")
# map kind -> (builtin_subdir, data_dir)
BUILTIN_KIND_DIRS = {
    "background": ("backgrounds", BG_DIR),
    "sound": ("sounds", SND_DIR),
}

DEFAULT_SETTINGS: Dict[str, Any] = {
    "refresh_interval_ms": 5000,
    "request_timeout_s": 1.2,
    "pause_polling_when_hidden": True,
    "destroy_coins_when_paused": False,
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
    "braiins": {
        "prefer_rest": True,
        "rest_scheme": "http",  # http|https (auto will try both)
        "rest_port": 80,
        "rest_username": "",
        "rest_password": "",
        "papi_port": 4028,
    },
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

    # Canonical schema for new installs.
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS dashboard_devices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            ip TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL,
            sort_order INTEGER NOT NULL DEFAULT 0,
            poll_type TEXT NOT NULL DEFAULT 'http',
            config_json TEXT,
            last_seen TEXT,
            last_poll TEXT,
            online INTEGER NOT NULL DEFAULT 0,
            last_error TEXT,
            last_info_json TEXT
        );
        """
    )

    def _table_info():
        cur.execute("PRAGMA table_info(dashboard_devices);")
        rows = cur.fetchall()
        cols = {r[1] for r in rows}  # (cid, name, type, notnull, dflt_value, pk)
        types = {r[1]: (r[2] or "") for r in rows}
        return cols, types

    def _add_col(col_name: str, ddl: str) -> None:
        cols, _types = _table_info()
        if col_name in cols:
            return
        cur.execute(ddl)

    # Protocol-aware polling.
    _add_col("poll_type", "ALTER TABLE dashboard_devices ADD COLUMN poll_type TEXT NOT NULL DEFAULT 'http';")

    # Optional per-device config JSON (credentials, ports, etc.)
    _add_col("config_json", "ALTER TABLE dashboard_devices ADD COLUMN config_json TEXT;")

    # Dashboard health/status fields. These are used by the polling loop; if they
    # are missing, the dashboard can 500 with "no such column".
    _add_col("last_seen", "ALTER TABLE dashboard_devices ADD COLUMN last_seen TEXT;")
    _add_col("last_poll", "ALTER TABLE dashboard_devices ADD COLUMN last_poll TEXT;")
    _add_col("online", "ALTER TABLE dashboard_devices ADD COLUMN online INTEGER NOT NULL DEFAULT 0;")
    _add_col("last_error", "ALTER TABLE dashboard_devices ADD COLUMN last_error TEXT;")
    _add_col("last_info_json", "ALTER TABLE dashboard_devices ADD COLUMN last_info_json TEXT;")

    # Cleanup migration for the historical missing-comma bug.
    # Old schema could yield a column type like: "TEXT\n            last_seen TEXT".
    cols, types = _table_info()
    cfg_type = str(types.get("config_json") or "")
    if cfg_type and ("\n" in cfg_type or "last_seen" in cfg_type.lower()):
        tmp = "dashboard_devices__rebuild"
        cur.execute(f"DROP TABLE IF EXISTS {tmp};")
        cur.execute(
            f"""
            CREATE TABLE {tmp} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                ip TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                sort_order INTEGER NOT NULL DEFAULT 0,
                poll_type TEXT NOT NULL DEFAULT 'http',
                config_json TEXT,
                last_seen TEXT,
                last_poll TEXT,
                online INTEGER NOT NULL DEFAULT 0,
                last_error TEXT,
                last_info_json TEXT
            );
            """
        )

        expected = [
            "id",
            "name",
            "ip",
            "created_at",
            "sort_order",
            "poll_type",
            "config_json",
            "last_seen",
            "last_poll",
            "online",
            "last_error",
            "last_info_json",
        ]

        cols, _types2 = _table_info()
        params = []
        select_parts = []
        now = _utcnow_iso()
        for c in expected:
            if c in cols:
                select_parts.append(c)
            else:
                if c == "online":
                    select_parts.append("0")
                elif c == "created_at":
                    select_parts.append("?")
                    params.append(now)
                else:
                    select_parts.append("NULL")

        cur.execute(
            f"INSERT INTO {tmp} ({', '.join(expected)}) "
            f"SELECT {', '.join(select_parts)} FROM dashboard_devices;",
            params,
        )

        cur.execute("DROP TABLE dashboard_devices;")
        cur.execute(f"ALTER TABLE {tmp} RENAME TO dashboard_devices;")

        # Keep AUTOINCREMENT sequence sane.
        try:
            cur.execute("SELECT COALESCE(MAX(id), 0) AS mx FROM dashboard_devices;")
            row = cur.fetchone()
            try:
                mx = int(row["mx"])  # sqlite3.Row
            except Exception:
                mx = int(row[0]) if row else 0
            cur.execute(
                "INSERT INTO sqlite_sequence(name, seq) VALUES('dashboard_devices', ?) "
                "ON CONFLICT(name) DO UPDATE SET seq=excluded.seq;",
                (mx,),
            )
        except Exception:
            pass

    # Normalize any empty values for older installs.
    cur.execute("UPDATE dashboard_devices SET poll_type='http' WHERE poll_type IS NULL OR TRIM(poll_type)='';")
    cur.execute("UPDATE dashboard_devices SET online=0 WHERE online IS NULL;")

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

def _seed_builtin_assets() -> None:
    """
    Copy any files from builtin_assets/{sounds,backgrounds} into the corresponding
    data/dashboard_assets/{sounds,backgrounds} folder, and register them in dashboard_assets.
    Safe to run multiple times.
    """
    _ensure_dirs()  # ensures BG_DIR / SND_DIR exist :contentReference[oaicite:7]{index=7}

    conn = db._get_conn()
    cur = conn.cursor()

    for kind, (subdir, out_dir) in BUILTIN_KIND_DIRS.items():
        src_dir = os.path.join(BUILTIN_ASSET_ROOT, subdir)
        if not os.path.isdir(src_dir):
            continue

        seeded_ids: list[int] = []

        for name in sorted(os.listdir(src_dir)):
            if name.startswith("."):
                continue
            src_path = os.path.join(src_dir, name)
            if not os.path.isfile(src_path):
                continue

            # Keep a stable filename in data/ (nice for humans + deterministic seeding)
            safe_name = os.path.basename(name).replace(" ", "_")
            dst_path = os.path.join(out_dir, safe_name)

            # Copy only if missing (don't clobber user-modified versions)
            if not os.path.exists(dst_path):
                os.makedirs(out_dir, exist_ok=True)
                shutil.copyfile(src_path, dst_path)

            # Register in DB if missing
            mime = mimetypes.guess_type(dst_path)[0] or ("audio/wav" if kind == "sound" else "application/octet-stream")
            size = os.path.getsize(dst_path)

            # avoid duplicates by (kind, filename) or (kind, orig_name)
            cur.execute(
                """
                SELECT id FROM dashboard_assets
                WHERE kind=? AND (filename=? OR orig_name=?)
                LIMIT 1;
                """,
                (kind, safe_name, name),
            )
            row = cur.fetchone()
            if row:
                seeded_ids.append(int(row["id"]))
                continue

            cur.execute(
                """
                INSERT INTO dashboard_assets (kind, filename, orig_name, mime, size_bytes, created_at, active)
                VALUES (?, ?, ?, ?, ?, ?, 0);
                """,
                (kind, safe_name, name, mime, size, _utcnow_iso()),
            )
            seeded_ids.append(int(cur.lastrowid))

        # If nothing is active for this kind yet, activate the first seeded one
        if seeded_ids:
            cur.execute("SELECT id FROM dashboard_assets WHERE kind=? AND active=1 LIMIT 1;", (kind,))
            has_active = cur.fetchone() is not None
            if not has_active:
                first_id = seeded_ids[0]
                cur.execute("UPDATE dashboard_assets SET active=0 WHERE kind=?;", (kind,))
                cur.execute("UPDATE dashboard_assets SET active=1 WHERE id=?;", (first_id,))

    conn.commit()
    conn.close()

# Ensure tables exist at import time (keeps app.py changes minimal).
_ensure_tables()
_seed_builtin_assets()

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



def _parse_device_cfg(device_row: Dict[str, Any]) -> Dict[str, Any]:
    """Parse per-device config JSON from DB row safely."""
    raw = device_row.get("config_json")
    if not raw:
        return {}
    try:
        if isinstance(raw, (dict, list)):
            return raw if isinstance(raw, dict) else {"raw": raw}
        return json.loads(raw) if isinstance(raw, str) else {}
    except Exception:
        return {}


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


def _extract_bracket_fields(raw: str, keys: List[str]) -> Dict[str, str]:
    """Extract fields formatted like Key[Value] from Avalon 'estats' blobs.

    Nano 3S (and some Avalon firmware) embeds many telemetry values inside a large
    string (often within the MM ID0 field) rather than emitting them as key=value
    pairs, so the regular cgminer comma/equals parser won't see them.
    """
    out: Dict[str, str] = {}
    if not raw:
        return out
    for k in keys:
        m = re.search(rf"\b{k}\[([^\]]+)\]", raw)
        if m:
            out[k] = m.group(1).strip()
    return out


def _sane_temp(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    # Some devices report "missing" temps as absurd values (e.g. -273).
    if x <= -200:
        return None
    return x


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
        v_raw = _cgminer_query(ip, "version", timeout_s)

        s_raw = _cgminer_query(ip, "summary", timeout_s)

        e_raw = _cgminer_query(ip, "estats", timeout_s)


        v_secs = _parse_cgminer_sections(v_raw)

        s_secs = _parse_cgminer_sections(s_raw)

        e_secs = _parse_cgminer_sections(e_raw)

        ver = _pick_first(v_secs, "PROD") or {}
        summ = _pick_first(s_secs, "Elapsed") or {}
        stats = _pick_first(e_secs, "STATS") or {}

        # Nano 3S-style telemetry: fields like OTemp[75], TAvg[80], FanR[21%]
        br = _extract_bracket_fields(
            e_raw,
            ["ITemp", "OTemp", "HBITemp", "HBOTemp", "TAvg", "TMax", "MTavg", "MTmax", "FanR", "Fan1", "Fan2", "Fan3", "Fan4", "Ver", "Power", "Pwr", "PWR", "POW", "Watts", "Watt", "Pout", "POUT", "VIN", "VIn", "Vin", "IIN", "IIn", "Iin", "PS"]
        )
        for k, v in br.items():
            stats.setdefault(k, v)


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

        chip_temp = _sane_temp(num(stats.get("TAvg") or stats.get("MTavg") or stats.get("HBOTemp") or stats.get("HBITemp")))
        out_temp = _sane_temp(num(stats.get("OTemp") or stats.get("HBOTemp")))
        board_temp = _sane_temp(num(stats.get("HBITemp")))
        in_temp = _sane_temp(num(stats.get("ITemp")))


        # power (W): not always emitted as a first-class field, but some firmwares embed it in estats.
        def _num_from_any(v: Any) -> Optional[float]:
            if v is None:
                return None
            if isinstance(v, (int, float)):
                return float(v)
            s = str(v).strip()
            if not s:
                return None
            # strip common unit suffixes
            s = s.replace("W", "").replace("w", "").replace("V", "").replace("v", "").replace("A", "").replace("a", "")
            s = s.replace("mV", "").replace("mA", "")
            # keep first numeric token
            mm = re.search(r"[-+]?[0-9]*\.?[0-9]+", s)
            if not mm:
                return None
            try:
                return float(mm.group(0))
            except Exception:
                return None

        power_w: Optional[float] = None

        # Avalon embeds power telemetry in PS[...] on some firmwares.
        # Example: PS[0 0 27440 4 0 3730 131]  -> last value ≈ watts (matches on-device ~130W).
        if power_w is None:
            ps = stats.get("PS")
            if ps is not None:
                nums = re.findall(r"[-+]?[0-9]*\.?[0-9]+", str(ps))
                if nums:
                    try:
                        w_guess = float(nums[-1])
                        # Avalon Q reports higher wattage than Nano 3S; allow a larger ceiling.
                        ps_max = 5000 if str(ver.get("MODEL") or "").strip().upper() == "Q" or "avalon q" in str(ver.get("PROD") or "").lower() else 500
                        if 10 <= w_guess <= ps_max:
                            power_w = w_guess
                    except Exception:
                        pass

        # direct power keys (best case)
        for k in ("Power", "Pwr", "PWR", "POW", "Watts", "Watt", "Pout", "POUT"):
            pv = _num_from_any(stats.get(k))
            if pv is not None and pv > 0:
                power_w = pv
                break

        # if we didn't find watts directly, try voltage/current (may be in mV/mA or V/A)
        if power_w is None:
            vin = None
            iin = None
            for vk in ("VIN", "VIn", "Vin"):
                vin = _num_from_any(stats.get(vk))
                if vin is not None:
                    break
            for ik in ("IIN", "IIn", "Iin"):
                iin = _num_from_any(stats.get(ik))
                if iin is not None:
                    break

            if vin is not None and iin is not None and vin > 0 and iin > 0:
                # heuristics: if values look like milli-units, scale down
                v_volts = vin / 1000.0 if vin > 200 else vin
                i_amps = iin / 1000.0 if iin > 20 else iin
                power_w = v_volts * i_amps






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
        prod = (ver.get("PROD") or "Avalon").strip()
        model = (ver.get("MODEL") or "").strip()
        # Avoid duplicate names like "Avalon Nano3s Nano3s"
        if model and (model.lower() == prod.lower() or model.lower() in prod.lower()):
            device_model = prod
        else:
            device_model = (f"{prod} {model}").strip() if model else prod

        # If version response is sparse, try estats Ver[...] (e.g. "Nano3s-25061101_...")
        if (not device_model or device_model.lower() == "avalon") and stats.get("Ver"):
            vv = str(stats.get("Ver"))
            base = vv.split("-", 1)[0].strip()
            if base:
                if base.lower().startswith("nano3s"):
                    device_model = "Avalon Nano 3S"
                else:
                    device_model = f"Avalon {base}"
        # Cosmetic normalization for Nano 3S naming
        if device_model and device_model.lower() == "avalon nano3s":
            device_model = "Avalon Nano 3S"

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
            "power": power_w,
            "temp": chip_temp,
            "outTemp": out_temp,
            "boardTemp": board_temp,
            "inTemp": in_temp,

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



# ---- Braiins OS / BOSminer support ----

_BRAIINS_TOKEN_CACHE: Dict[str, Dict[str, Any]] = {}  # ip -> {"token": str, "expires_at": float}


def _merge_braiins_cfg(device_cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge global settings defaults with per-device overrides."""
    cfg = {}
    try:
        s = _get_settings()
        if isinstance(s.get("braiins"), dict):
            cfg.update(s["braiins"])
    except Exception:
        pass
    if device_cfg and isinstance(device_cfg, dict):
        # allow either nested {"braiins": {...}} or top-level keys for convenience
        b = device_cfg.get("braiins")
        if isinstance(b, dict):
            cfg.update(b)
        for k in ("prefer_rest", "rest_scheme", "rest_port", "rest_username", "rest_password", "papi_port"):
            if k in device_cfg:
                cfg[k] = device_cfg[k]
    # normalize
    cfg["rest_port"] = int(cfg.get("rest_port") or 80)
    cfg["papi_port"] = int(cfg.get("papi_port") or 4028)
    if cfg.get("rest_scheme") not in ("http", "https"):
        cfg["rest_scheme"] = str(cfg.get("rest_scheme") or "http").lower()
    cfg["prefer_rest"] = bool(cfg.get("prefer_rest", True))
    return cfg


def _first_number(d: Any, keys: List[str]) -> Optional[float]:
    """Return first numeric value among candidate keys (case-insensitive), supporting nested dicts."""
    if not isinstance(d, dict):
        return None
    lower_map = {str(k).lower(): k for k in d.keys()}
    for k in keys:
        kk = str(k).lower()
        if kk in lower_map:
            v = d[lower_map[kk]]
            if isinstance(v, (int, float)):
                return float(v)
            if isinstance(v, str):
                try:
                    return float(v.strip())
                except Exception:
                    pass
            if isinstance(v, dict):
                # common nested measurement forms
                for subk in ("c", "value_c", "temperature_c", "temp_c", "value", "temperature", "temp"):
                    subv = v.get(subk)
                    if isinstance(subv, (int, float)):
                        return float(subv)
                    if isinstance(subv, str):
                        try:
                            return float(subv.strip())
                        except Exception:
                            pass
    return None


def _deep_find_numbers(obj: Any, key_hints: List[str], max_hits: int = 1) -> List[float]:
    """Recursively collect numeric values where key name includes any hint."""
    hits: List[float] = []
    hints = [h.lower() for h in key_hints]
    def walk(x: Any):
        nonlocal hits
        if len(hits) >= max_hits:
            return
        if isinstance(x, dict):
            for k, v in x.items():
                k_l = str(k).lower()
                if any(h in k_l for h in hints) and isinstance(v, (int, float, str, dict)):
                    num = _first_number({k: v}, [k])  # reuse
                    if num is not None:
                        hits.append(float(num))
                        if len(hits) >= max_hits:
                            return
                walk(v)
                if len(hits) >= max_hits:
                    return
        elif isinstance(x, list):
            for it in x:
                walk(it)
                if len(hits) >= max_hits:
                    return
    walk(obj)
    return hits


def _extract_temperature(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except Exception:
            return None
    if isinstance(value, dict):
        for k in ("c", "value_c", "temperature_c", "temp_c", "value", "temperature", "temp"):
            v = value.get(k)
            if isinstance(v, (int, float)):
                return float(v)
            if isinstance(v, str):
                try:
                    return float(v.strip())
                except Exception:
                    pass
    return None


def _bosminer_query(ip: str, command: str, timeout_s: float, port: int = 4028, req_id: int = 1) -> Dict[str, Any]:
    """BOSminer/Braiins PAPI: JSON command over TCP (usually port 4028)."""
    payload = json.dumps({"command": command, "id": req_id}) + "\n"
    buf = b""
    with socket.create_connection((ip, int(port)), timeout=timeout_s) as sock:
        sock.settimeout(timeout_s)
        sock.sendall(payload.encode("utf-8"))
        # Read until socket closes or we can parse JSON.
        while True:
            try:
                chunk = sock.recv(65536)
            except socket.timeout:
                break
            if not chunk:
                break
            buf += chunk
            if len(buf) > 2_000_000:
                break
    txt = buf.decode("utf-8", errors="replace").strip()
    # sometimes there may be extra lines; try last JSON object
    candidates = [t for t in txt.splitlines() if t.strip()]
    if not candidates:
        raise RuntimeError("Empty response")
    last = candidates[-1]
    try:
        data = json.loads(last)
    except Exception:
        # fallback to entire buffer
        data = json.loads(txt)
    if not isinstance(data, dict):
        raise RuntimeError("Unexpected response type")
    return data


def _probe_bosminer_papi(ip: str, timeout_s: float, cfg: Optional[Dict[str, Any]] = None) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    port = int((cfg or {}).get("papi_port") or 4028)
    try:
        data = _bosminer_query(ip, "fans", timeout_s, port=port, req_id=1)
        desc = None
        try:
            st = (data.get("STATUS") or [{}])[0]
            desc = st.get("Description") or st.get("Msg")
        except Exception:
            pass
        # Heuristic: BOSminer responses typically contain STATUS + the section matching the command.
        if "FANS" in data and "STATUS" in data:
            return True, {"description": desc, "port": port}, None
        return False, None, "unexpected response"
    except Exception as e:
        return False, None, str(e)


def _poll_bosminer_papi(ip: str, timeout_s: float, cfg: Optional[Dict[str, Any]] = None) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    port = int((cfg or {}).get("papi_port") or 4028)
    try:
        summary = _bosminer_query(ip, "summary", timeout_s, port=port, req_id=1)
        temps = _bosminer_query(ip, "temps", timeout_s, port=port, req_id=2)
        fans = _bosminer_query(ip, "fans", timeout_s, port=port, req_id=3)

        # summary parsing (BOSminer tends to mirror cgminer-ish keys)
        srow = None
        if isinstance(summary.get("SUMMARY"), list) and summary["SUMMARY"]:
            srow = summary["SUMMARY"][0]
        elif isinstance(summary.get("SUMMARY"), dict):
            srow = summary["SUMMARY"]
        else:
            srow = summary

        # hashrate: prefer GH/s if present; else MH/s fields.
        hr_gh = _first_number(srow, ["ghs_5s", "ghs", "ghs5s", "ghs_5", "ghs_15m", "ghs_av", "ghs_avg"])
        if hr_gh is None:
            mh = _first_number(srow, ["mhs 5s", "mhs_5s", "mhs5s", "mhs av", "mhs_av", "mhs_avg", "mhs"])
            if mh is not None:
                hr_gh = float(mh) / 1000.0

        power = _first_number(srow, ["power", "watts", "watt", "power_w", "power (w)"])
        if power is None:
            # some firmwares expose power under STATS/DEVS; try a quick fallback
            try:
                stats = _bosminer_query(ip, "stats", timeout_s, port=port, req_id=4)
                power = _first_number(stats, ["power", "watts", "power_w"])
            except Exception:
                power = None

        acc = _first_number(srow, ["accepted", "shares accepted", "accepted_shares"])
        rej = _first_number(srow, ["rejected", "shares rejected", "rejected_shares"])
        best = _first_number(srow, ["best share", "best_share", "bestshare", "best difficulty", "best_difficulty", "bestdiff"])

        # temps parsing: max chip + board
        chip_max = None
        board_max = None
        tlist = temps.get("TEMPS") if isinstance(temps, dict) else None
        if isinstance(tlist, list):
            chips = [float(t.get("Chip")) for t in tlist if isinstance(t, dict) and isinstance(t.get("Chip"), (int, float))]
            boards = [float(t.get("Board")) for t in tlist if isinstance(t, dict) and isinstance(t.get("Board"), (int, float))]
            chip_max = max(chips) if chips else None
            board_max = max(boards) if boards else None

        # fans parsing: avg speed % + rpm
        f_speed = None
        f_rpm = None
        flist = fans.get("FANS") if isinstance(fans, dict) else None
        if isinstance(flist, list):
            speeds = [float(f.get("Speed")) for f in flist if isinstance(f, dict) and isinstance(f.get("Speed"), (int, float))]
            rpms = [float(f.get("RPM")) for f in flist if isinstance(f, dict) and isinstance(f.get("RPM"), (int, float))]
            f_speed = sum(speeds) / len(speeds) if speeds else None
            f_rpm = sum(rpms) / len(rpms) if rpms else None

        info: Dict[str, Any] = {
            "type": "Braiins OS (BOSminer PAPI)",
            "papi_port": port,
            "hostname": None,
            "deviceModel": "BOSminer",
            "hashrate": hr_gh,
            "power": power,
            "temp": chip_max,
            "boardTemp": board_max,
            "fanspeed": f_speed,
            "fanrpm": f_rpm,
            "sharesAccepted": acc,
            "sharesRejected": rej,
            "bestDiff": best,
            "raw": {
                "summary": summary,
                "temps": temps,
                "fans": fans,
            },
        }

        return True, info, None
    except Exception as e:
        return False, None, str(e)


def _probe_braiins_rest(ip: str, timeout_s: float, cfg: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    """Detect Braiins OS REST API via /api/v1/miner/details over http/https.

    Older versions of this project probed /api/v1/version and treated any 401/403
    as a positive signal. That can false-positive on non-miner devices.

    /api/v1/miner/details is a Braiins OS specific endpoint (documented in the
    official OpenAPI spec) and is much less likely to collide.
    """
    pref_port = int(cfg.get("rest_port") or 80)
    ports = list(dict.fromkeys([pref_port, 80, 443]))
    schemes = ["https", "http"] if cfg.get("rest_scheme") == "https" else ["http", "https"]
    last_err: Optional[str] = None

    def looks_like_bos_details(js: object) -> bool:
        if not isinstance(js, dict):
            return False
        # keys from the OpenAPI sample payload
        expected = {"bos_version", "bos_mode", "miner_identity", "mac_address", "hostname", "serial_number", "uid"}
        return any(k in js for k in expected)

    for scheme in schemes:
        for port in ports:
            base = (
                f"{scheme}://{ip}:{port}"
                if (port and ((scheme == "http" and port != 80) or (scheme == "https" and port != 443)))
                else f"{scheme}://{ip}"
            )
            url = base + "/api/v1/miner/details"
            try:
                r = requests.get(url, timeout=timeout_s, verify=False)
                if r.status_code not in (200, 401, 403):
                    continue

                js = None
                try:
                    js = r.json()
                except Exception:
                    js = None

                if r.status_code == 200:
                    if looks_like_bos_details(js):
                        return True, {"base_url": base, "scheme": scheme, "port": port, "details": js}, None
                    # 200 but not the expected shape => not Braiins
                    continue

                # 401/403: still a good signal, but require at least JSON content-type
                # or a payload that matches the expected shape.
                ctype = (r.headers.get("content-type") or "").lower()
                if looks_like_bos_details(js) or ("json" in ctype):
                    return True, {"base_url": base, "scheme": scheme, "port": port, "details": js}, None

            except Exception as e:
                last_err = str(e)
                continue

    return False, None, last_err or "no response"


def _braiins_get_token(ip: str, base_url: str, cfg: Dict[str, Any], timeout_s: float) -> Optional[str]:
    user = str(cfg.get("rest_username") or "").strip()
    pw = str(cfg.get("rest_password") or "").strip()
    if not user or not pw:
        return None

    now = time.time()
    cached = _BRAIINS_TOKEN_CACHE.get(ip)
    if cached and cached.get("token") and float(cached.get("expires_at") or 0) > now + 10:
        return str(cached["token"])

    url = base_url + "/api/v1/auth/login"
    try:
        r = requests.post(url, json={"username": user, "password": pw}, timeout=timeout_s, verify=False)
        if r.status_code >= 400:
            return None
        js = r.json() if r.content else {}
        token = js.get("token") or js.get("access_token") or js.get("jwt")
        ttl = js.get("timeout_s") or js.get("expires_in") or 3600
        if token:
            _BRAIINS_TOKEN_CACHE[ip] = {"token": token, "expires_at": now + float(ttl)}
            return str(token)
        return None
    except Exception:
        return None


def _braiins_get_json(ip: str, base_url: str, path: str, cfg: Dict[str, Any], timeout_s: float) -> Tuple[Optional[Dict[str, Any]], Optional[int]]:
    headers = {"Accept": "application/json"}
    token = _braiins_get_token(ip, base_url, cfg, timeout_s)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    url = base_url + path
    r = requests.get(url, timeout=timeout_s, verify=False, headers=headers)
    if r.status_code >= 400:
        return None, r.status_code
    try:
        js = r.json()
    except Exception:
        return None, r.status_code
    return js if isinstance(js, dict) else {"raw": js}, r.status_code


def _poll_braiins_rest(
    ip: str,
    timeout_s: float,
    cfg: Dict[str, Any],
    rest_meta: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    """Poll Braiins OS Public REST API for stats/temps/fans."""
    try:
        if not rest_meta:
            ok, rest_meta, err = _probe_braiins_rest(ip, max(0.2, min(0.8, timeout_s)), cfg)
            if not ok:
                return False, None, err
        base = rest_meta.get("base_url") if rest_meta else None
        if not base:
            return False, None, "missing base_url"

        details, sc0 = _braiins_get_json(ip, base, "/api/v1/miner/details", cfg, timeout_s)

        stats, sc1 = _braiins_get_json(ip, base, "/api/v1/miner/stats", cfg, timeout_s)
        hashboards, sc2 = _braiins_get_json(ip, base, "/api/v1/miner/hw/hashboards", cfg, timeout_s)
        cooling, sc3 = _braiins_get_json(ip, base, "/api/v1/cooling/state", cfg, timeout_s)
        # hostname (nice-to-have)
        netinfo, _sc4 = _braiins_get_json(ip, base, "/api/v1/network/", cfg, max(0.3, min(0.9, timeout_s)))

        # If we have no creds and got 401s, return a stub that tells the UI what's missing.
        if (stats is None and sc1 in (401, 403)) or (hashboards is None and sc2 in (401, 403)):
            info = {
                "type": "Braiins OS (REST)",
                "deviceModel": "Braiins OS",
                "hostname": (netinfo or {}).get("hostname"),
                "authRequired": True,
                "rest_base": base,
            }
            return True, info, None

        
        # Prefer identity info from /api/v1/miner/details when available
        if isinstance(details, dict):
            hn = details.get("hostname")
            if hn:
                info["hostname"] = hn

            # Try to pull a real hardware model (Antminer model, etc.) from miner_identity
            model = None
            mid = details.get("miner_identity")
            if isinstance(mid, dict):
                # Prefer keys that look like model/name
                for k in ("model", "hardware_model", "hardwareModel", "name", "type", "variant"):
                    v = mid.get(k)
                    if isinstance(v, str) and v.strip():
                        model = v.strip()
                        break
                if model is None:
                    # fall back to first non-empty string value
                    for v in mid.values():
                        if isinstance(v, str) and v.strip():
                            model = v.strip()
                            break

            # If we didn't get it from miner_identity, try sticker_hashrate/container fields
            if model is None:
                v = details.get("platform")
                if isinstance(v, (str, int)):
                    model = str(v)

            if model:
                info["deviceModel"] = model

            info.setdefault("raw", {}).setdefault("details", details)

        # Temperatures: use max highest_chip_temp and board_temp if available
        chip_vals: List[float] = []
        board_vals: List[float] = []
        hb_list = (hashboards or {}).get("hashboards")
        if isinstance(hb_list, list):
            for hb in hb_list:
                if not isinstance(hb, dict):
                    continue
                chip_vals.append(_extract_temperature(hb.get("highest_chip_temp")) or None)
                # prefer board_temp else outlet temp
                board_vals.append(_extract_temperature(hb.get("board_temp")) or _extract_temperature(hb.get("highest_outlet_temp")) or None)
            chip_vals = [v for v in chip_vals if isinstance(v, (int, float))]
            board_vals = [v for v in board_vals if isinstance(v, (int, float))]
        if chip_vals:
            info["temp"] = max(chip_vals)
        if board_vals:
            info["boardTemp"] = max(board_vals)

        # Fans: average target_speed_ratio (0-1) -> %
        fanspeed = None
        fanrpm = None
        fans_list = (cooling or {}).get("fans")
        if isinstance(fans_list, list):
            ratios = []
            rpms = []
            for f in fans_list:
                if not isinstance(f, dict):
                    continue
                r = _first_number(f, ["target_speed_ratio", "speed_ratio", "target_speed", "speed"])
                if r is not None:
                    ratios.append(r)
                rpm = _first_number(f, ["rpm", "speed_rpm"])
                if rpm is not None:
                    rpms.append(rpm)
            if ratios:
                # if ratios look like 0-1, convert; if already 0-100, keep
                avg = sum(ratios)/len(ratios)
                fanspeed = avg*100.0 if avg <= 1.5 else avg
            if rpms:
                fanrpm = sum(rpms)/len(rpms)
        if fanspeed is not None:
            info["fanspeed"] = fanspeed
        if fanrpm is not None:
            info["fanrpm"] = fanrpm

        # Hashrate / power / shares / best diff from stats (best-effort heuristics)
        ms = (stats or {}).get("miner_stats") if isinstance(stats, dict) else None
        ps = (stats or {}).get("pool_stats") if isinstance(stats, dict) else None
        pws = (stats or {}).get("power_stats") if isinstance(stats, dict) else None

        # hashrate
        hr = None
        for src in (ms, ps, stats):
            if hr is not None:
                break
            if isinstance(src, dict):
                hr = _first_number(src, ["hashrate_ghs", "hashrate_gh", "ghs", "ghs_5s", "hashrate_ths", "ths", "hashrate"])
        if hr is not None:
            # If it looks like TH/s, convert to GH/s
            if hr < 200 and (_deep_find_numbers(stats, ["ths"], max_hits=1) or _deep_find_numbers(stats, ["hashrate_th"], max_hits=1)):
                info["hashrate"] = hr * 1000.0
            else:
                # heuristic: if "hashrate" is in H/s, it's huge; convert to GH
                if hr > 1e6:
                    info["hashrate"] = hr / 1e9
                else:
                    info["hashrate"] = hr

        # power
        pw = None
        for src in (pws, ms, stats):
            if pw is not None:
                break
            if isinstance(src, dict):
                pw = _first_number(src, ["power_w", "power", "watts", "watt", "consumption_w"])
        if pw is not None:
            info["power"] = pw

        # shares + rejected
        acc = None
        rej = None
        best = None
        for src in (ps, ms, stats):
            if isinstance(src, dict):
                if acc is None:
                    acc = _first_number(src, ["accepted", "shares_accepted", "accepted_shares", "sharesAccepted"])
                if rej is None:
                    rej = _first_number(src, ["rejected", "shares_rejected", "rejected_shares", "sharesRejected"])
                if best is None:
                    best = _first_number(src, ["best_difficulty", "best_diff", "bestshare", "best_share", "best"])
        if acc is None:
            hits = _deep_find_numbers(stats, ["accepted"], max_hits=1)
            acc = hits[0] if hits else None
        if rej is None:
            hits = _deep_find_numbers(stats, ["rejected"], max_hits=1)
            rej = hits[0] if hits else None
        if best is None:
            hits = _deep_find_numbers(stats, ["best"], max_hits=1)
            best = hits[0] if hits else None

        if acc is not None:
            info["sharesAccepted"] = acc
        if rej is not None:
            info["sharesRejected"] = rej
        if best is not None:
            info["bestDiff"] = best

        return True, info, None
    except Exception as e:
        return False, None, str(e)


def _looks_like_http_miner_payload(data: object) -> bool:
    """Heuristic: True if JSON looks like a supported miner HTTP payload.

    Used to keep LAN scan results from listing random devices that happen to
    expose JSON endpoints. We require both an identity hint and at least one
    mining/telemetry metric.
    """
    if not isinstance(data, dict):
        return False

    # Identity hints
    ident_ok = any(k in data for k in (
        "deviceModel", "ASICModel", "minerModel", "model", "hwModel", "hardwareModel",
    ))

    # Telemetry / mining-ish hints
    metric_ok = any(k in data for k in (
        "hashRate", "hashrate", "hashRate_1m", "hashRate_10m", "hashRate_1h",
        "power", "temp", "boardTemp", "chipTemp", "vrmTemp",
        "fanspeed", "fanrpm",
        "sharesAccepted", "sharesRejected", "bestDiff", "foundBlocks",
        "uptimeSeconds", "macAddr",
    ))

    return bool(ident_ok and metric_ok)


def _looks_like_supported_miner(detected: str, info: object) -> bool:
    """Return True if a (detected, info) pair appears to be a miner we support."""
    d = (detected or "").strip().lower()
    if d in ("avalon_cgminer", "bosminer_papi", "braiins_rest"):
        return True
    if d in ("http", "bitaxe", "nerdqaxe"):
        return _looks_like_http_miner_payload(info)
    return False


def _fetch_system_info(
    ip: str,
    timeout_s: float,
    poll_type: str = "auto",
    device_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str], str]:
    pt = (poll_type or "auto").strip().lower()

    # Explicit Braiins OS REST polling
    if pt in ("braiins", "braiins_rest", "bos_rest", "bos_api", "rest"):
        cfg = _merge_braiins_cfg(device_cfg)
        ok, info, err = _poll_braiins_rest(ip, timeout_s, cfg, None)
        return ok, info, err, "braiins_rest"

    # Explicit BOSminer/Braiins legacy PAPI polling
    if pt in ("bosminer", "bosminer_papi", "braiins_papi", "papi"):
        cfg = _merge_braiins_cfg(device_cfg)
        ok, info, err = _poll_bosminer_papi(ip, timeout_s, cfg)
        return ok, info, err, "bosminer_papi"

    # Explicit Avalon polling
    if pt in ("avalon", "avalon_q", "cgminer", "avalon_cgminer"):
        ok, info, err = _poll_avalon_q(ip, timeout_s)
        return ok, info, err, "avalon_cgminer"

    # Explicit HTTP polling (BitAxe/NerdQAxe style)
    if pt in ("http", "bitaxe", "nerdqaxe"):
        url = f"http://{ip}/api/system/info"
        try:
            r = requests.get(url, timeout=timeout_s)
            r.raise_for_status()
            data = r.json()
            if not _looks_like_http_miner_payload(data):
                return False, None, "Not a supported miner HTTP API", "http"
            return True, data, None, "http"
        except Exception as e:
            return False, None, str(e), "http"

    # Auto-detect:
    # 1) Try the most specific protocols first (TCP miners), then REST, then HTTP.
    cfg = _merge_braiins_cfg(device_cfg)
    quick = max(0.15, min(0.45, timeout_s))

    # BOSminer/Braiins PAPI (JSON over TCP/4028)
    ok_papi, _meta_p, err_papi = _probe_bosminer_papi(ip, quick, cfg)
    if ok_papi:
        ok_full, info_p, err_full = _poll_bosminer_papi(ip, timeout_s, cfg)
        return ok_full, info_p, err_full, "bosminer_papi"

    # Avalon cgminer (TCP/4028 pipe protocol)
    ok_probe, _ver, err_a = _probe_avalon_q(ip, quick)
    if ok_probe:
        ok_full, info_a, err_full = _poll_avalon_q(ip, timeout_s)
        return ok_full, info_a, err_full, "avalon_cgminer"

    # Braiins OS REST (HTTP/HTTPS)
    ok_rest, rest_meta, err_rest = _probe_braiins_rest(ip, quick, cfg)
    if ok_rest:
        ok_full, info_r, err_full = _poll_braiins_rest(ip, timeout_s, cfg, rest_meta)
        return ok_full, info_r, err_full, "braiins_rest"

    # Finally try BitAxe-style HTTP
    url = f"http://{ip}/api/system/info"
    try:
        r = requests.get(url, timeout=timeout_s)
        r.raise_for_status()
        data = r.json()
        if not _looks_like_http_miner_payload(data):
            raise RuntimeError("Not a supported miner HTTP API")
        return True, data, None, "http"
    except Exception as e:
        extras = []
        if err_rest:
            extras.append(f"rest probe: {err_rest}")
        if err_papi:
            extras.append(f"bosminer probe: {err_papi}")
        if err_a:
            extras.append(f"avalon probe: {err_a}")
        extra = (" (" + "; ".join(extras) + ")") if extras else ""
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
        cfg = _parse_device_cfg(d)
        ok, info, err, detected = _fetch_system_info(d["ip"], timeout, poll_type=pt, device_cfg=cfg)
        poll_update = detected if ok and detected in ("http", "avalon_cgminer", "braiins_rest", "bosminer_papi") else None
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


@router.get("/debug")
def api_debug():
    """Return a lightweight diagnostic snapshot to help troubleshoot deployments.

    This endpoint is intentionally read-only and avoids returning any secrets.
    It is safe to share screenshots/output when reporting issues.
    """
    info: Dict[str, Any] = {
        "ok": True,
        "db_path": getattr(db, "DB_PATH", None),
        "cwd": os.getcwd(),
        "base_dir": _BASE_DIR,
        "asset_root": ASSET_ROOT,
        "bg_dir": BG_DIR,
        "snd_dir": SND_DIR,
    }

    try:
        conn = db._get_conn()
        cur = conn.cursor()

        cur.execute("PRAGMA database_list;")
        info["sqlite_databases"] = [
            {"seq": r[0], "name": r[1], "file": r[2]} for r in cur.fetchall()
        ]

        cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
        tables = [r[0] if not hasattr(r, "keys") else r["name"] for r in cur.fetchall()]
        info["tables"] = tables

        def table_info(table: str) -> Dict[str, Any]:
            out: Dict[str, Any] = {"columns": []}
            try:
                cur.execute(f"PRAGMA table_info({table});")
                cols = []
                for row in cur.fetchall():
                    # row = (cid, name, type, notnull, dflt_value, pk)
                    cols.append(
                        {
                            "name": row[1],
                            "type": row[2],
                            "notnull": bool(row[3]),
                            "default": row[4],
                            "pk": bool(row[5]),
                        }
                    )
                out["columns"] = cols
                cur.execute(f"SELECT COUNT(*) FROM {table};")
                out["row_count"] = int(cur.fetchone()[0])
            except Exception as e:
                out["error"] = f"{type(e).__name__}: {e}"
            return out

        # Focus on the tables relevant to the dashboard and benchmarks.
        for t in ("dashboard_devices", "dashboard_settings", "dashboard_assets", "benchmark_runs", "profiles"):
            if t in tables:
                info[t] = table_info(t)

        conn.close()
    except Exception as e:
        info["ok"] = False
        info["error"] = f"{type(e).__name__}: {e}"

    return info


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
        if not _looks_like_supported_miner(detected, info):
            return None
        # attach a few convenient fields
        if isinstance(info, dict):
            hostname = info.get("hostname") or info.get("host") or None
            model = info.get("deviceModel") or info.get("ASICModel") or None
        else:
            hostname, model = None, None
        return {"ip": ip, "hostname": hostname, "model": model, "detected": detected, "info": info}

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
