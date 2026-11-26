# miner_api.py
"""
Firmware abstraction for the benchmark engine.

The engine was written against the AxeOS (ESP-Miner) HTTP API:
  GET   /api/system/info
  GET   /api/system/asic
  PATCH /api/system            {"coreVoltage", "frequency"}
  POST  /api/system/restart

This module adds a second backend for Harlo OS (Harlo Labs firmware, tested
against the 0.2.101 API shape) and exposes both through the same interface.
Harlo OS responses are normalized to AxeOS field names so the rest of the
engine does not need to know which firmware it is talking to.

Harlo OS endpoints used (shape taken from the HashFleet Harlo driver,
github.com/SMWARREN/HashFleet, which derives request bodies from the web UI
embedded in the Harlo OS 0.2.101 firmware image):
  GET  /api/info        static info + "profile" (asic_count, stock/min/max)
  GET  /api/telemetry   live readings (hashrate_ghs, asic_temp_c, ...)
  PUT  /api/tuning      {"frequency_mhz": int, "core_mv": int} (both required)
  POST /api/reboot
Harlo OS answers unknown or rejected requests with {"ok": false, "output": "..."}.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict, Optional

import requests


AXEOS = "axeos"
HARLO_OS = "harlo_os"

# Small cores per ASIC, used to compute an expected hashrate when the firmware
# does not report one (expected GH/s = MHz * small_cores * asic_count / 1000).
SMALL_CORES_BY_CHIP: Dict[str, int] = {
    "BM1397": 672,
    "BM1366": 894,
    "BM1368": 1276,
    "BM1370": 2040,
}


class MinerAPIError(RuntimeError):
    pass


def normalize_base_url(ip_or_url: str) -> str:
    base = (ip_or_url or "").strip().rstrip("/")
    if not base.startswith("http://") and not base.startswith("https://"):
        base = "http://" + base
    return base


def _env_flag(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _num(val: Any) -> Optional[float]:
    if isinstance(val, bool):
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        try:
            return float(val.strip())
        except ValueError:
            return None
    return None


def _first_num(src: Dict[str, Any], *keys: str) -> Optional[float]:
    for k in keys:
        v = _num(src.get(k))
        if v is not None:
            return v
    return None


# ---------------------------------------------------------------------------
# AxeOS (original behavior, unchanged)
# ---------------------------------------------------------------------------


class AxeOSClient:
    firmware = AXEOS
    firmware_label = "AxeOS"

    def __init__(self, base_url: str):
        self.base_url = normalize_base_url(base_url)

    def get_system_info(self, timeout: float = 10) -> Dict[str, Any]:
        r = requests.get(f"{self.base_url}/api/system/info", timeout=timeout)
        r.raise_for_status()
        return r.json()

    def get_asic_info(self, timeout: float = 10) -> Dict[str, Any]:
        r = requests.get(f"{self.base_url}/api/system/asic", timeout=timeout)
        r.raise_for_status()
        return r.json()

    def apply_settings(self, core_voltage: int, frequency: int, timeout: float = 10) -> None:
        payload = {"coreVoltage": core_voltage, "frequency": frequency}
        r = requests.patch(f"{self.base_url}/api/system", json=payload, timeout=timeout)
        r.raise_for_status()

    # AxeOS only applies V/F after a restart.
    needs_restart_after_apply = True

    def restart(self, timeout: float = 10) -> None:
        r = requests.post(f"{self.base_url}/api/system/restart", timeout=timeout)
        r.raise_for_status()

    def verify_settings(self, core_voltage: int, frequency: int) -> Optional[str]:
        return None


# ---------------------------------------------------------------------------
# Harlo OS
# ---------------------------------------------------------------------------


class HarloOSClient:
    firmware = HARLO_OS
    firmware_label = "Harlo OS"

    def __init__(self, base_url: str):
        self.base_url = normalize_base_url(base_url)
        self._info_cache: Optional[Dict[str, Any]] = None
        self._info_lock = threading.Lock()
        # Harlo OS applies tuning live. A reboot after every step is optional
        # (HARLO_REBOOT_AFTER_TUNING=1) for people who want a clean start per point.
        self.needs_restart_after_apply = _env_flag("HARLO_REBOOT_AFTER_TUNING", False)

    # --- raw calls ---

    def _get(self, path: str, timeout: float) -> Dict[str, Any]:
        r = requests.get(f"{self.base_url}{path}", timeout=timeout)
        data = _json_or_none(r)
        if isinstance(data, dict) and data.get("ok") is False:
            raise MinerAPIError(f"Harlo OS {path}: {data.get('output') or 'rejected'}")
        r.raise_for_status()
        if not isinstance(data, dict):
            raise MinerAPIError(f"Harlo OS {path}: response is not a JSON object")
        return data

    def _send(self, method: str, path: str, body: Optional[Dict[str, Any]], timeout: float) -> Dict[str, Any]:
        r = requests.request(method, f"{self.base_url}{path}", json=body, timeout=timeout)
        data = _json_or_none(r)
        if isinstance(data, dict) and data.get("ok") is False:
            raise MinerAPIError(f"Harlo OS {method} {path} rejected: {data.get('output') or 'no reason given'}")
        r.raise_for_status()
        return data if isinstance(data, dict) else {}

    def info(self, timeout: float = 10, refresh: bool = False) -> Dict[str, Any]:
        with self._info_lock:
            if self._info_cache is None or refresh:
                self._info_cache = self._get("/api/info", timeout)
            return self._info_cache

    def telemetry(self, timeout: float = 10) -> Dict[str, Any]:
        return self._get("/api/telemetry", timeout)

    # --- profile helpers ---

    def profile(self) -> Dict[str, Any]:
        try:
            p = self.info().get("profile")
        except Exception:
            return {}
        return p if isinstance(p, dict) else {}

    def chip_model(self, combined: Optional[Dict[str, Any]] = None) -> Optional[str]:
        combined = combined or {}
        prof = self.profile()
        for src in (prof, combined):
            for k in ("asic_model", "ASICModel", "chip", "asic"):
                v = src.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip().upper()
        if "bm1373_counters" in combined:
            return "BM1373"
        if "bm1370_counters" in combined:
            return "BM1370"
        pid = str(prof.get("id") or prof.get("name") or "").lower()
        if any(s in pid for s in ("b30", "thor", "nexus-s1")):
            return "BM1373"
        if any(s in pid for s in ("gamma", "nerdqaxe", "nerdoctaxe", "bitfortun", "naja")):
            return "BM1370"
        return None

    def limits(self) -> Dict[str, Optional[int]]:
        p = self.profile()

        def as_int(k: str) -> Optional[int]:
            v = _num(p.get(k))
            return int(v) if v is not None else None

        return {
            "frequency_min": as_int("frequency_min_mhz"),
            "frequency_max": as_int("frequency_max_mhz"),
            "core_min": as_int("core_min_mv"),
            "core_max": as_int("core_max_mv"),
            "stock_frequency": as_int("stock_frequency_mhz"),
            "stock_core": as_int("stock_core_mv"),
            "power_supply_w": as_int("power_supply_default_w"),
        }

    # --- AxeOS-shaped interface ---

    def get_system_info(self, timeout: float = 10) -> Dict[str, Any]:
        tel = self.telemetry(timeout)
        try:
            info = self.info(timeout)
        except Exception:
            info = {}
        return self.normalize(info, tel)

    def get_asic_info(self, timeout: float = 10) -> Dict[str, Any]:
        lim = self.limits()
        prof = self.profile()
        chip = self.chip_model()
        out: Dict[str, Any] = {
            "ASICModel": chip,
            "deviceModel": prof.get("name") or prof.get("id"),
            "asicCount": int(_num(prof.get("asic_count")) or 1),
        }
        if lim["stock_frequency"] is not None:
            out["defaultFrequency"] = lim["stock_frequency"]
        if lim["stock_core"] is not None:
            out["defaultVoltage"] = lim["stock_core"]
        # Extra (non-AxeOS) keys so the engine can respect firmware limits.
        out["harloLimits"] = lim
        return out

    def apply_settings(self, core_voltage: int, frequency: int, timeout: float = 10) -> None:
        self._send(
            "PUT",
            "/api/tuning",
            {"frequency_mhz": int(frequency), "core_mv": int(core_voltage)},
            timeout,
        )

    def restart(self, timeout: float = 10) -> None:
        self._send("POST", "/api/reboot", None, timeout)

    def verify_settings(self, core_voltage: int, frequency: int) -> Optional[str]:
        """Return None if the miner reports the requested targets, else a reason."""
        try:
            tel = self.telemetry()
        except Exception as e:
            return f"could not read telemetry to verify tuning: {e}"
        f = _first_num(tel, "target_frequency_mhz", "frequency_mhz")
        v = _first_num(tel, "target_core_mv")
        problems = []
        if f is not None and abs(f - frequency) > 1:
            problems.append(f"frequency target is {f:g} MHz, expected {frequency}")
        if v is not None and abs(v - core_voltage) > 1:
            problems.append(f"core target is {v:g} mV, expected {core_voltage}")
        return "; ".join(problems) or None

    def normalize(self, info: Dict[str, Any], tel: Dict[str, Any]) -> Dict[str, Any]:
        """Map Harlo OS /api/info + /api/telemetry to AxeOS /api/system/info keys."""
        c: Dict[str, Any] = dict(info)
        c.update(tel)
        prof = c.get("profile") if isinstance(c.get("profile"), dict) else self.profile()

        out: Dict[str, Any] = {
            "firmwareFamily": HARLO_OS,
            "_harloRaw": tel,
        }

        hr = _first_num(c, "hashrate_ghs")
        if hr is not None:
            out["hashRate"] = hr

        temp = _first_num(c, "asic_temp_c")
        if temp is not None:
            out["temp"] = temp
        vr = _first_num(c, "vr_temp_c")
        if vr is not None:
            out["vrTemp"] = vr

        power = _first_num(c, "device_power_w", "asic_rail_power_w")
        if power is not None:
            out["power"] = power

        vin = _first_num(c, "input_voltage_v", "input_voltage_mv")
        if vin is not None:
            # AxeOS reports input voltage in mV.
            out["voltage"] = vin * 1000.0 if vin < 100 else vin
        else:
            # Not every board exposes an input voltage; tell the engine to skip
            # the input-voltage checks instead of failing the run.
            out["voltage"] = None
            out["_inputVoltageUnavailable"] = True

        freq = _first_num(c, "target_frequency_mhz", "frequency_mhz", "frequency")
        if freq is not None:
            out["frequency"] = int(round(freq))
        pll = _first_num(c, "active_pll_mhz")
        if pll is not None:
            out["actualFrequency"] = pll

        core_target = _first_num(c, "target_core_mv", "core_mv")
        if core_target is not None:
            out["coreVoltage"] = int(round(core_target))
        core_actual = _first_num(c, "core_voltage_mv")
        if core_actual is not None:
            out["coreVoltageActual"] = core_actual

        fans = c.get("fans") if isinstance(c.get("fans"), list) else []
        fan1 = next(
            (f for f in fans if isinstance(f, dict) and int(_num(f.get("channel")) or 0) == 1),
            fans[0] if fans and isinstance(fans[0], dict) else {},
        )
        fan_pct = _first_num(fan1, "percent", "applied_percent", "manual_percent")
        if fan_pct is None:
            fan_pct = _first_num(c, "fan_duty_pct", "fanspeed")
        if fan_pct is not None:
            out["fanspeed"] = fan_pct
        rpm = _first_num(fan1, "rpm")
        if rpm is not None:
            out["fanrpm"] = rpm
        if len(fans) > 1 and isinstance(fans[1], dict):
            rpm2 = _first_num(fans[1], "rpm")
            if rpm2 is not None:
                out["fan2rpm"] = rpm2
        tt = _first_num(fan1, "target_c")
        if tt is not None:
            out["temptarget"] = tt
        if isinstance(fan1.get("mode"), str):
            out["autofanspeed"] = 1 if fan1["mode"].lower() == "auto" else 0

        # Shares: sum over pool slots, fall back to top-level counters.
        acc = rej = 0
        seen = False
        for p in c.get("pools") or []:
            if not isinstance(p, dict):
                continue
            a = _num(p.get("accepted_shares"))
            r = _num(p.get("rejected_shares"))
            if a is not None:
                acc += int(a)
                seen = True
            if r is not None:
                rej += int(r)
                seen = True
        if not seen:
            a = _num(c.get("accepted_shares"))
            r = _num(c.get("rejected_shares"))
            if a is not None and r is not None:
                acc, rej, seen = int(a), int(r), True
        if seen:
            out["sharesAccepted"] = acc
            out["sharesRejected"] = rej

        err = _first_num(c, "error_percentage", "errorPercentage", "hw_error_pct", "asic_error_pct")
        if err is not None:
            out["errorPercentage"] = err

        chip = self.chip_model(c)
        if chip:
            out["ASICModel"] = chip
        asic_count = _num(prof.get("asic_count")) if prof else None
        if asic_count:
            out["asicCount"] = int(asic_count)
        small = SMALL_CORES_BY_CHIP.get(chip or "")
        if small:
            out["smallCoreCount"] = small
        if small and asic_count and freq:
            out["expectedHashrate"] = freq * small * asic_count / 1000.0

        name = c.get("miner_name") or c.get("name")
        if isinstance(name, str) and name:
            out["hostname"] = name
        if prof:
            out["boardVersion"] = prof.get("name") or prof.get("id")
            out["deviceModel"] = prof.get("name") or prof.get("id")
        ver = c.get("firmware_version") or c.get("version") or c.get("firmware")
        if ver:
            out["version"] = ver
            out["axeOSVersion"] = f"Harlo OS {ver}"
        up = _first_num(c, "uptime_seconds")
        if up is not None:
            out["uptimeSeconds"] = int(up)
        best = c.get("best_share_session")
        if best is not None:
            out["bestSessionDiff"] = best
        best_all = c.get("best_share_all_time")
        if best_all is not None:
            out["bestDiff"] = best_all
        if c.get("asic_paused") is not None:
            out["asicPaused"] = bool(c.get("asic_paused"))
        if c.get("fault_notice_pending"):
            out["faultNotice"] = " - ".join(
                str(x) for x in (c.get("fault_notice_title"), c.get("fault_notice_reason")) if x
            )
        return out


def _json_or_none(r: requests.Response) -> Any:
    try:
        return r.json()
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def looks_like_harlo(data: Any) -> bool:
    if not isinstance(data, dict):
        return False
    fw = str(data.get("firmware") or "").lower()
    if fw.startswith("harlo"):
        return True
    if "hashrate_ghs" not in data and "target_frequency_mhz" not in data:
        return False
    return any(
        k in data
        for k in ("asic_temp_c", "device_power_w", "asic_paused", "miner_name", "target_core_mv")
    )


def looks_like_axeos(data: Any) -> bool:
    return isinstance(data, dict) and (
        "hashRate" in data or "ASICModel" in data or "coreVoltage" in data
    )


_detect_cache: Dict[str, str] = {}
_detect_lock = threading.Lock()


def detect_firmware(base_url: str, timeout: float = 5.0, use_cache: bool = True) -> str:
    """
    Return AXEOS or HARLO_OS. Set MINER_FIRMWARE=axeos|harlo_os to force a backend.
    AxeOS is probed first so existing setups behave exactly as before.
    """
    forced = (os.environ.get("MINER_FIRMWARE") or "").strip().lower()
    if forced in (AXEOS, HARLO_OS):
        return forced

    base = normalize_base_url(base_url)
    if use_cache:
        with _detect_lock:
            if base in _detect_cache:
                return _detect_cache[base]

    result: Optional[str] = None
    try:
        r = requests.get(f"{base}/api/system/info", timeout=timeout)
        if r.ok and looks_like_axeos(_json_or_none(r)):
            result = AXEOS
    except requests.exceptions.RequestException:
        pass

    if result is None:
        for path in ("/api/info", "/api/telemetry"):
            try:
                r = requests.get(f"{base}{path}", timeout=timeout)
                if r.ok and looks_like_harlo(_json_or_none(r)):
                    result = HARLO_OS
                    break
            except requests.exceptions.RequestException:
                continue

    if result is None:
        # Unknown or unreachable: keep the original AxeOS behavior (and its errors).
        return AXEOS

    with _detect_lock:
        _detect_cache[base] = result
    return result


def get_client(base_url: str, timeout: float = 5.0):
    fw = detect_firmware(base_url, timeout=timeout)
    if fw == HARLO_OS:
        return HarloOSClient(base_url)
    return AxeOSClient(base_url)


def fetch_system_info_any(ip_or_url: str, timeout: float = 5.0) -> Optional[Dict[str, Any]]:
    """For the dashboard: try Harlo OS and return AxeOS-shaped info, or None."""
    base = normalize_base_url(ip_or_url)
    try:
        r = requests.get(f"{base}/api/info", timeout=timeout)
        if not (r.ok and looks_like_harlo(_json_or_none(r))):
            return None
    except requests.exceptions.RequestException:
        return None
    try:
        return HarloOSClient(base).get_system_info(timeout=timeout)
    except Exception:
        return None
