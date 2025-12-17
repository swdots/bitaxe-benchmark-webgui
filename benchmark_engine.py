# benchmark_engine.py
from __future__ import annotations

from dataclasses import dataclass, asdict, fields
from enum import Enum
from typing import Optional, List, Dict, Any, Callable
import threading
import time
import json
import os
from datetime import datetime
import math  # NEW

import requests


class DeviceType(str, Enum):
    GAMMA_602 = "gamma_602"
    NERDQAXE_PP = "nerdqaxe_pp"
    OTHER = "other"


@dataclass
class BenchmarkConfig:
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
    
    # New: optional error-rate threshold (%). If None, we fall back to 2.0.
    error_rate_warn_threshold: Optional[float] = 2.0
    
    # NEW: sweep mode – "adaptive" (current behavior) or "grid"
    sweep_mode: str = "adaptive"


# Built-in safe-ish overrides per device type
BUILTIN_PROFILE_OVERRIDES: Dict[DeviceType, Dict[str, Any]] = {
    DeviceType.GAMMA_602: {
        "max_power": 40,
        "max_temp": 66,
        "max_vr_temp": 86,
        "min_input_voltage": 4800,
        "max_input_voltage": 5500,
    },
    DeviceType.NERDQAXE_PP: {
        # Adjust as you learn NerdQaxe++ behavior
        "max_power": 115,
        "max_temp": 70,
        "max_vr_temp": 90,
        "min_input_voltage": 11800,
        "max_input_voltage": 12500,
    },
    DeviceType.OTHER: {},
}


def make_builtin_config(device_type: DeviceType) -> BenchmarkConfig:
    overrides = BUILTIN_PROFILE_OVERRIDES.get(device_type, {})
    return BenchmarkConfig(device_type=device_type, **overrides)


def identify_device(bitaxe_ip: str, timeout: float = 5.0) -> Dict[str, Any]:
    """
    Query /api/system/info and /api/system/asic to:
      - auto-identify device type (Gamma vs NerdQaxe++ vs Other)
      - provide a suggested config
      - return device metadata and current starting conditions
    """
    base = bitaxe_ip.strip()
    if not base.startswith("http://") and not base.startswith("https://"):
        base = "http://" + base

    system_info: Dict[str, Any] = {}
    asic_info: Dict[str, Any] = {}

    # /api/system/info
    try:
        r = requests.get(f"{base}/api/system/info", timeout=timeout)
        r.raise_for_status()
        system_info = r.json()
    except Exception:
        system_info = {}

    # /api/system/asic
    try:
        r2 = requests.get(f"{base}/api/system/asic", timeout=timeout)
        r2.raise_for_status()
        asic_info = r2.json()
    except Exception:
        asic_info = {}

    # Build a combined label string from known fields to classify device
    labels: List[str] = []
    for src in (asic_info, system_info):
        if not isinstance(src, dict):
            continue
        for key in (
            "deviceModel",
            "boardVersion",
            "model",
            "deviceName",
            "boardName",
            "ASICModel",
        ):
            val = src.get(key)
            if isinstance(val, str):
                labels.append(val.lower())

    label_str = " ".join(labels)

    dtype: DeviceType = DeviceType.OTHER
    if "nerdqaxe" in label_str:
        dtype = DeviceType.NERDQAXE_PP
    elif "gamma" in label_str or "bitaxe" in label_str or "602" in label_str:
        dtype = DeviceType.GAMMA_602
    else:
        # fallback heuristic: multi-ASIC devices are more likely NerdQaxe++
        asic_count = None
        if isinstance(asic_info, dict):
            asic_count = asic_info.get("asicCount")
        if asic_count is None and isinstance(system_info, dict):
            asic_count = system_info.get("asicCount")
        try:
            if asic_count and int(asic_count) >= 4:
                dtype = DeviceType.NERDQAXE_PP
        except Exception:
            pass

    builtin_cfg = make_builtin_config(dtype)

    # Determine starting/default voltage & frequency from device
    start_v = None
    start_f = None
    # Prefer current live settings from /info
    if isinstance(system_info, dict):
        start_v = system_info.get("coreVoltage")
        start_f = system_info.get("frequency")
    # Fallback to default values from /asic
    if start_v is None and isinstance(asic_info, dict):
        start_v = asic_info.get("defaultVoltage")
    if start_f is None and isinstance(asic_info, dict):
        start_f = asic_info.get("defaultFrequency")

    cfg_dict = asdict(builtin_cfg)
    if isinstance(start_v, (int, float)):
        cfg_dict["initial_voltage"] = int(start_v)
    if isinstance(start_f, (int, float)):
        cfg_dict["initial_frequency"] = int(start_f)

    # Device metadata to show in UI / notes
    device_meta = {
        "asicModel": (
            (isinstance(asic_info, dict) and asic_info.get("ASICModel"))
            or (isinstance(system_info, dict) and system_info.get("ASICModel"))
        ),
        "deviceModel": (isinstance(asic_info, dict) and asic_info.get("deviceModel")) or None,
        "boardVersion": isinstance(system_info, dict) and system_info.get("boardVersion"),
        "asicCount": (
            (isinstance(asic_info, dict) and asic_info.get("asicCount"))
            or (isinstance(system_info, dict) and system_info.get("asicCount"))
        ),
        "smallCoreCount": isinstance(system_info, dict) and system_info.get("smallCoreCount"),
        "firmwareVersion": isinstance(system_info, dict) and system_info.get("version"),
        "axeOSVersion": isinstance(system_info, dict) and system_info.get("axeOSVersion"),
        "hostname": isinstance(system_info, dict) and system_info.get("hostname"),
    }

    # Starting / current runtime readings from /info
    starting = {}
    if isinstance(system_info, dict):
        for key in (
            "coreVoltage",
            "coreVoltageActual",
            "frequency",
            "temp",
            "temp2",
            "vrTemp",
            "power",
            "voltage",
            "hashRate",
            "expectedHashrate",
            "fanspeed",
            "fanrpm",
            "temptarget",
            "errorPercentage",
        ):
            if key in system_info:
                starting[key] = system_info[key]

    return {
        "deviceType": dtype.value,
        "config": cfg_dict,
        "deviceInfo": device_meta,
        "starting": starting,
        "rawInfo": {"systemInfo": system_info, "asicInfo": asic_info},
    }


class BenchmarkStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BenchmarkRunner:
    """
    Encapsulates a single benchmark run against one device.
    Designed to be run in a background thread so the web API stays responsive.
    """

    def __init__(
        self,
        run_id: str,
        bitaxe_ip: str,
        config: BenchmarkConfig,
        on_finish: Optional[Callable[["BenchmarkRunner", Optional[str]], None]] = None,
    ):
        self.run_id = run_id
        self.bitaxe_ip_raw = bitaxe_ip.strip()
        if self.bitaxe_ip_raw.startswith("http://") or self.bitaxe_ip_raw.startswith("https://"):
            self.base_url = self.bitaxe_ip_raw
        else:
            self.base_url = f"http://{self.bitaxe_ip_raw}"

        self.config = config
        self.on_finish = on_finish

        self.status: BenchmarkStatus = BenchmarkStatus.PENDING
        # Global progress 0–100 for the whole run
        self.progress: float = 0.0
        self.current_voltage: int = config.initial_voltage
        self.current_frequency: int = config.initial_frequency
        self.results: List[Dict[str, Any]] = []
        self.last_sample: Optional[Dict[str, Any]] = None
        self.error_reason: Optional[str] = None

        # info discovered from device
        self.default_voltage: Optional[int] = None
        self.default_frequency: Optional[int] = None
        self.small_core_count: Optional[int] = None
        self.asic_count: Optional[int] = None

        self.best_hashrate: Optional[float] = None
        self.best_efficiency: Optional[float] = None

        self.start_time: Optional[str] = None
        self.end_time: Optional[str] = None

        # ETA-related
        self.eta_seconds: Optional[float] = None
        self._start_dt: Optional[datetime] = None
        self._approx_total_iterations: int = 1
        self._iteration_index: int = 0

        # Narrative + recent point history
        self.status_detail: Optional[str] = None

        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

    # --- public API ---

    def start(self) -> None:
        if self._thread:
            raise RuntimeError("Benchmark already started")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def cancel(self) -> None:
        self._stop.set()
        

    def to_dict(self, include_results: bool = False) -> Dict[str, Any]:
        with self._lock:
            data: Dict[str, Any] = {
                "runId": self.run_id,
                "bitaxeIp": self.bitaxe_ip_raw,
                "status": self.status.value,
                "progress": self.progress,
                "currentVoltage": self.current_voltage,
                "currentFrequency": self.current_frequency,
                "deviceType": self.config.device_type.value,
                "startTime": self.start_time,
                "endTime": self.end_time,
                "errorReason": self.error_reason,
                "bestHashrate": self.best_hashrate,
                "bestEfficiency": self.best_efficiency,
                "lastSample": self.last_sample,
                "config": asdict(self.config),
                "etaSeconds": self.eta_seconds,
                "statusDetail": self.status_detail,
                # last 10 completed test points
                "recentResults": self.results[-10:] if self.results else [],
            }
            if include_results:
                data["results"] = self.results
            return data

    # --- ETA helpers ---

    def _estimate_total_iterations(self) -> int:
        cfg = self.config
        if cfg.frequency_increment <= 0 or cfg.voltage_increment <= 0:
            return 1
        vol_steps = max(
            1, (cfg.max_allowed_voltage - cfg.initial_voltage) // cfg.voltage_increment + 1
        )
        freq_steps = max(
            1, (cfg.max_allowed_frequency - cfg.initial_frequency) // cfg.frequency_increment + 1
        )
        est = vol_steps * freq_steps
        return max(1, min(est, 64))  # cap so ETAs don't explode for huge ranges
        
    def _refine_eta_iterations_based_on_headroom(self) -> None:
        """
        Refine _approx_total_iterations based on how quickly we are burning
        through *fan* headroom and, once the fan is saturated, thermal headroom.

        Model:
          - While the automatic fan curve is still below ~100%, the board is
            mostly holding ~60°C and the *fan percentage* is the best proxy
            for "how hard we're pushing".
          - Once the fan is basically maxed out, further V/F increases show up
            as a rising chip temperature, so we switch to using temp slope.

        This is deliberately conservative: we only ever LOWER the estimate.
        """
        points = self.results
        # Need a few points to get a meaningful slope
        if len(points) < 3:
            return

        cfg = self.config
        original_est = self._approx_total_iterations
        if original_est <= len(points) + 1:
            # Nothing to refine; we're already at or below what we think is left
            return

        # Look at the last few completed points
        window_size = min(len(points), 6)
        window = points[-window_size:]

        def get_series(key: str):
            vals = [p.get(key) for p in window if isinstance(p.get(key), (int, float))]
            return vals if len(vals) >= 2 else None

        fan_series = get_series("fanSpeed")
        temp_series = get_series("averageTemperature")

        # If we don't have fan data, fall back to old-style temp-only refinement.
        if not fan_series:
            if not temp_series or not cfg.max_temp:
                return
            start_t, end_t = temp_series[0], temp_series[-1]
            steps = len(temp_series) - 1
            temp_slope = (end_t - start_t) / steps if steps > 0 else 0.0
            if temp_slope <= 0.1:  # °C per step
                return
            headroom = max(0.0, float(cfg.max_temp) - end_t)
            est_remaining = headroom / temp_slope
            predicted_total = len(points) + max(1, int(math.ceil(est_remaining)))
            predicted_total = max(len(points) + 1, min(predicted_total, original_est))
            self._approx_total_iterations = predicted_total
            return

        # --- Phase selection based on fan saturation ---
        last_fan = fan_series[-1]
        first_fan = fan_series[0]
        steps_fan = len(fan_series) - 1
        fan_slope = (last_fan - first_fan) / steps_fan if steps_fan > 0 else 0.0

        FAN_SAT_THRESHOLD = 95.0  # "basically maxed" on these boards
        predicted_total = original_est  # default: no change

        if last_fan < FAN_SAT_THRESHOLD:
            # Fan is still ramping up toward 100%: use fan slope as the primary limiter.
            # Ignore tiny / noisy slopes.
            if fan_slope <= 0.25:  # % per iteration
                return

            headroom_pct = max(0.0, 99.0 - last_fan)  # leave a little buffer below 100%
            est_steps_to_sat = headroom_pct / fan_slope

            # Add a small fixed tail for the "fan-saturated but not yet thermally limited"
            # region. This is heuristic but keeps ETA from dropping to zero immediately
            # when we first hit ~100% fan.
            est_remaining = est_steps_to_sat + 3.0

            predicted_total = len(points) + max(1, int(math.ceil(est_remaining)))
        else:
            # Fan is at or near 100%: now temperature is the interesting limiter.
            if not temp_series or not cfg.max_temp:
                # No useful temp info -> don't touch ETA.
                return

            start_t, end_t = temp_series[0], temp_series[-1]
            steps_t = len(temp_series) - 1
            temp_slope = (end_t - start_t) / steps_t if steps_t > 0 else 0.0

            # Require a meaningful positive slope; if temps are flat or falling,
            # we're not clearly marching toward the thermal ceiling.
            if temp_slope <= 0.1:
                return

            headroom_deg = max(0.0, float(cfg.max_temp) - end_t)
            est_steps_to_thermal = headroom_deg / temp_slope

            predicted_total = len(points) + max(1, int(math.ceil(est_steps_to_thermal)))

        # Clamp so we only shrink (or keep) the estimate; never grow it,
        # and never say "we're done" before at least one more iteration.
        predicted_total = max(len(points) + 1, min(predicted_total, original_est))
        self._approx_total_iterations = predicted_total

    def _update_eta(self, iteration_index: int, sample_index: int, total_samples: int) -> None:
        """
        Also updates global progress based on iteration + intra-iteration position.
        """
        if not self._start_dt:
            return
        elapsed = (datetime.utcnow() - self._start_dt).total_seconds()
        if elapsed <= 0:
            return

        frac_iter = iteration_index + (sample_index / max(1, total_samples))
        overall_fraction = frac_iter / max(1, self._approx_total_iterations)
        if overall_fraction <= 0:
            return

        total_est = elapsed / overall_fraction
        self.eta_seconds = max(0.0, total_est - elapsed)
        # Global progress 0–100
        self.progress = max(0.0, min(100.0, overall_fraction * 100.0))

    # --- narrative helpers ---

    def _describe_error_reason(self, code: str) -> str:
        mapping = {
            "CANCELLED": "cancelled by request",
            "SYSTEM_INFO_FAILURE": "unable to read /api/system/info from device",
            "MISSING_TEMP_OR_VOLTAGE": "missing temperature or input voltage readings",
            "TEMPERATURE_BELOW_5": "chip temperature reading below 5°C (sensor/boot issue)",
            "CHIP_TEMP_EXCEEDED": "chip temperature exceeded configured max",
            "VR_TEMP_EXCEEDED": "VR temperature exceeded configured max",
            "INPUT_VOLTAGE_BELOW_MIN": "input voltage dropped below configured minimum",
            "INPUT_VOLTAGE_ABOVE_MAX": "input voltage exceeded configured maximum",
            "MISSING_HASHRATE_OR_POWER": "missing hashrate or power readings",
            "POWER_CONSUMPTION_EXCEEDED": "power consumption above configured limit",
            "NO_DATA_COLLECTED": "no valid samples collected",
            "ZERO_HASHRATE": "average hashrate was zero",
            "ERROR_RATE_TOO_HIGH": "ASIC error rate exceeded configured threshold",
        }
        return mapping.get(code, code.replace("_", " ").lower())

    def _build_status_detail(
        self,
        iteration_index: int,
        current_voltage: int,
        current_frequency: int,
        prev_hashrate_ok: Optional[bool],
        prev_error_reason: Optional[str],
        prev_avg_hashrate: Optional[float],
        prev_avg_temp: Optional[float],
        prev_efficiency: Optional[float],
        prev_avg_error_pct: Optional[float],
        prev_expected_hashrate: Optional[float],
    ) -> str:
        cfg = self.config
        total_iters = self._approx_total_iterations
        iter_label = f"step {iteration_index + 1} of ~{total_iters}"

        # Why this step?
        if iteration_index == 0:
            reason = "starting baseline sweep from initial profile"
        elif prev_error_reason:
            reason = (
                f"previous run stopped due to {self._describe_error_reason(prev_error_reason)}; "
                "this is the next safe point in the sweep"
            )
        elif prev_hashrate_ok is True:
            reason = (
                "previous run passed thermal and hashrate checks; "
                f"increasing frequency by +{cfg.frequency_increment} MHz"
            )
        elif prev_hashrate_ok is False:
            # Decide whether we backed off due to error-rate or hashrate
            threshold = cfg.error_rate_warn_threshold
            if threshold is None:
                threshold = 2.0

            if (
                prev_avg_error_pct is not None
                and prev_avg_error_pct > threshold
            ):
                # Error-rate driven backoff
                reason = (
                    f"previous run's ASIC error rate was {prev_avg_error_pct:.2f}% "
                    f"(threshold {threshold:.2f}%); "
                    f"raising core voltage by +{cfg.voltage_increment} mV and backing off frequency"
                )
            else:
                # Hashrate-driven backoff
                if (
                    prev_avg_hashrate is not None
                    and prev_expected_hashrate is not None
                    and prev_expected_hashrate > 0
                ):
                    pct_of_target = 100.0 * prev_avg_hashrate / prev_expected_hashrate
                    reason = (
                        f"previous run's hashrate was {prev_avg_hashrate:.1f} GH/s "
                        f"vs expected {prev_expected_hashrate:.1f} GH/s "
                        f"({pct_of_target:.1f}% of target); "
                        f"raising core voltage by +{cfg.voltage_increment} mV and backing off frequency"
                    )
                else:
                    # Fallback if we didn't know expected hashrate
                    reason = (
                        "previous run's hashrate was below expected; "
                        f"raising core voltage by +{cfg.voltage_increment} mV and backing off frequency"
                    )


        else:
            reason = "continuing sweep based on previous measurements"

        if prev_avg_hashrate is not None and prev_avg_temp is not None and prev_efficiency is not None:
            reason += (
                f" (last point: {prev_avg_hashrate:.1f} GH/s, "
                f"{prev_avg_temp:.1f} °C, {prev_efficiency:.2f} J/TH)"
            )

        # What’s next (if stable)?
        next_if_stable_freq = None
        if current_frequency + cfg.frequency_increment <= cfg.max_allowed_frequency:
            next_if_stable_freq = current_frequency + cfg.frequency_increment

        if next_if_stable_freq is not None:
            next_clause = (
                f"next target {next_if_stable_freq} MHz @ {current_voltage} mV if stable"
            )
        else:
            # At max freq; further steps would be voltage-driven or we're done
            next_clause = "this is near the configured limit; will stop or adjust voltage after this point"

        return (
            f"Testing {current_frequency} MHz @ {current_voltage} mV "
            f"({iter_label}) — {reason}. {next_clause}."
        )

    # --- main run loop ---

    def _run(self) -> None:
        self.start_time = datetime.utcnow().isoformat()
        self._start_dt = datetime.utcnow()
        self.status = BenchmarkStatus.RUNNING
        result_path: Optional[str] = None

        prev_hashrate_ok: Optional[bool] = None
        prev_error_reason: Optional[str] = None
        prev_avg_hashrate: Optional[float] = None
        prev_avg_temp: Optional[float] = None
        prev_efficiency: Optional[float] = None
        prev_avg_error_pct: Optional[float] = None
        prev_expected_hashrate: Optional[float] = None 

        try:
            self._validate_config()
            self._fetch_default_settings()

            self._approx_total_iterations = self._estimate_total_iterations()
            iteration_index = 0

            current_voltage = self.config.initial_voltage
            current_frequency = self.config.initial_frequency

            while (
                current_voltage <= self.config.max_allowed_voltage
                and current_frequency <= self.config.max_allowed_frequency
            ):
                if self._stop.is_set():
                    self.status = BenchmarkStatus.CANCELLED
                    self.error_reason = "Cancelled by user"
                    with self._lock:
                        self.status_detail = (
                            "Benchmark cancelled by user; cleaning up and restoring device settings."
                        )
                    break

                # Update current V/F and narrative before this iteration
                self.current_voltage = current_voltage
                self.current_frequency = current_frequency

                with self._lock:
                    self.status_detail = self._build_status_detail(
                        iteration_index=iteration_index,
                        current_voltage=current_voltage,
                        current_frequency=current_frequency,
                        prev_hashrate_ok=prev_hashrate_ok,
                        prev_error_reason=prev_error_reason,
                        prev_avg_hashrate=prev_avg_hashrate,
                        prev_avg_temp=prev_avg_temp,
                        prev_efficiency=prev_efficiency,
                        prev_avg_error_pct=prev_avg_error_pct,
                        prev_expected_hashrate=prev_expected_hashrate, 
                    )

                self._set_system_settings(current_voltage, current_frequency)

                # expose current iteration index for ETA updates
                self._iteration_index = iteration_index

                (
                    avg_hashrate,
                    avg_temp,
                    efficiency_jth,
                    hashrate_ok,
                    avg_vr_temp,
                    error_reason,
                    extras,
                ) = self._benchmark_iteration(current_voltage, current_frequency)

                iteration_index += 1

                prev_hashrate_ok = hashrate_ok
                prev_error_reason = error_reason
                prev_avg_hashrate = avg_hashrate
                prev_avg_temp = avg_temp
                prev_efficiency = efficiency_jth

                if self._stop.is_set():
                    self.status = BenchmarkStatus.CANCELLED
                    self.error_reason = "Cancelled by user"
                    with self._lock:
                        self.status_detail = (
                            "Benchmark cancelled by user; cleaning up and restoring device settings."
                        )
                    break

                if error_reason is not None:
                    # hit some thermal/power/safety limit
                    self.error_reason = error_reason
                    # If we have no results at all, treat this as a hard failure.
                    # If we already collected valid points, treat it as a
                    # graceful stop at the safety limit.
                    if not self.results:
                        # full-run failure: nothing usable was collected
                        self.status = BenchmarkStatus.FAILED
                    self._stop.set()                               # ← optional, but harmless
                    with self._lock:
                        self.status_detail = (
                            f"Stopping benchmark: {self._describe_error_reason(error_reason)}."
                        )
                    break

                if (
                    avg_hashrate is not None
                    and avg_temp is not None
                    and efficiency_jth is not None
                ):
                    avg_power = extras.get("avgPower") if extras else None
                    avg_fan_pct = extras.get("avgFanPct") if extras else None
                    avg_fan_rpm = extras.get("avgFanRpm") if extras else None
                    hashrate_domains = extras.get("hashrateDomains") if extras else None
                    avg_err_pct = extras.get("avgErrorPercentage") if extras else None
                    expected_hash = extras.get("expectedHashrate") if extras else None
                    reject_rate = extras.get("rejectRate") if extras else None
                    shares_acc = extras.get("sharesAccepted") if extras else None
                    shares_rej = extras.get("sharesRejected") if extras else None
                    shares_total = extras.get("sharesTotal") if extras else None

                    # Remember these for the next step's narrative
                    prev_avg_error_pct = avg_err_pct
                    prev_expected_hashrate = expected_hash

                    result: Dict[str, Any] = {
                        "coreVoltage": current_voltage,
                        "frequency": current_frequency,
                        "averageHashRate": avg_hashrate,
                        "averageTemperature": avg_temp,
                        "efficiencyJTH": efficiency_jth,
                    }
                    if avg_power is not None:
                        result["averagePower"] = avg_power
                    if avg_vr_temp is not None:
                        result["averageVRTemp"] = avg_vr_temp
                    if avg_fan_pct is not None:
                        result["fanSpeed"] = avg_fan_pct     # picked up by UI as fanSpeed/fan_pct/fanspeed
                    if avg_fan_rpm is not None:
                        result["fanRPM"] = avg_fan_rpm       # picked up by UI as fanRPM/fan_rpm/fanrpm
                    if avg_err_pct is not None:
                        result["errorPercentage"] = avg_err_pct
                    if reject_rate is not None:
                        result["rejectRate"] = reject_rate
                    if shares_acc is not None:
                        result["sharesAccepted"] = shares_acc
                    if shares_rej is not None:
                        result["sharesRejected"] = shares_rej
                    if shares_total is not None:
                        result["sharesTotal"] = shares_total
                    if hashrate_domains is not None:
                        # full per-ASIC / per-domain stats
                        result["hashrateDomains"] = hashrate_domains

                    self.results.append(result)
                    
                    # Refine ETA based on real cooling / error behaviour
                    self._refine_eta_iterations_based_on_headroom()

                    # update bests
                    if self.best_hashrate is None or avg_hashrate > self.best_hashrate:
                        self.best_hashrate = avg_hashrate
                    if self.best_efficiency is None or efficiency_jth < self.best_efficiency:
                        self.best_efficiency = efficiency_jth

                    # same decision logic as original script
                    if hashrate_ok:
                        # If hashrate is good, try increasing frequency
                        if (
                            current_frequency + self.config.frequency_increment
                            <= self.config.max_allowed_frequency
                        ):
                            current_frequency += self.config.frequency_increment
                        else:
                            # can't increase further; we're done
                            with self._lock:
                                self.status_detail = (
                                    "Benchmark completed: reached maximum configured frequency "
                                    "with acceptable hashrate and thermals."
                                )
                            break
                    else:
                        # If hashrate is not good, go back one frequency step and increase voltage
                        if (
                            current_voltage + self.config.voltage_increment
                            <= self.config.max_allowed_voltage
                        ):
                            current_voltage += self.config.voltage_increment
                            current_frequency -= self.config.frequency_increment
                            if current_frequency < self.config.min_allowed_frequency:
                                current_frequency = self.config.min_allowed_frequency
                        else:
                            # We've hit the voltage ceiling and still don't have an acceptable point.
                            # Decide whether to classify this as an error-rate failure or a hashrate failure.
                            threshold = (
                                self.config.error_rate_warn_threshold
                                if self.config.error_rate_warn_threshold is not None
                                else 2.0
                            )

                            if avg_err_pct is not None and avg_err_pct > threshold:                                
                                # Persistent high error-rate: stop
                                self.error_reason = "ERROR_RATE_TOO_HIGH"
                                msg = (
                                    "Benchmark stopping: voltage ceiling reached with ASIC error rate above "
                                    f"threshold ({avg_err_pct:.2f}% > {threshold:.2f}%)."
                                )
                            else:
                                # Treat as hashrate-based ceiling
                                msg = (
                                    "Benchmark stopping: voltage ceiling reached with suboptimal "
                                    "hashrate; not pushing further."
                                )

                            with self._lock:
                                self.status_detail = msg
                            break
                else:
                    # no data / zero hashrate etc – stop
                    with self._lock:
                        self.status_detail = (
                            "Benchmark stopping: no valid data collected at this point."
                        )
                    break

            # After loop finishes, reset device
            self._reset_to_best_or_default()
            if self.results:
                result_path = self._write_results_json()
                if self.status == BenchmarkStatus.RUNNING:
                    # Only mark completed if we never transitioned to FAILED / CANCELLED
                    self.status = BenchmarkStatus.COMPLETED
                    with self._lock:
                        if not self.status_detail:
                            self.status_detail = (
                                "Benchmark completed; restored device to best-performing settings."
                            )

            else:
                if self.status != BenchmarkStatus.CANCELLED:
                    self.status = BenchmarkStatus.FAILED
                    if self.error_reason is None:
                        self.error_reason = "No valid benchmarking results"
                    with self._lock:
                        self.status_detail = (
                            f"Benchmark failed: {self._describe_error_reason(self.error_reason)}."
                        )

        except Exception as e:
            self.status = BenchmarkStatus.FAILED
            self.error_reason = f"Unexpected error: {e}"
            with self._lock:
                self.status_detail = f"Benchmark failed due to unexpected error: {e}."
            try:
                self._reset_to_best_or_default()
            except Exception:
                pass
        finally:
            self.end_time = datetime.utcnow().isoformat()
            if self.on_finish:
                try:
                    self.on_finish(self, result_path)
                except Exception:
                    pass

    # --- helpers: config & device talking ---

    def _validate_config(self) -> None:
        cfg = self.config
        if cfg.initial_voltage > cfg.max_allowed_voltage:
            raise ValueError(
                f"Initial voltage {cfg.initial_voltage} exceeds max allowed {cfg.max_allowed_voltage}"
            )
        if cfg.initial_voltage < cfg.min_allowed_voltage:
            raise ValueError(
                f"Initial voltage {cfg.initial_voltage} below min allowed {cfg.min_allowed_voltage}"
            )
        if cfg.initial_frequency > cfg.max_allowed_frequency:
            raise ValueError(
                f"Initial frequency {cfg.initial_frequency} exceeds max allowed {cfg.max_allowed_frequency}"
            )
        if cfg.initial_frequency < cfg.min_allowed_frequency:
            raise ValueError(
                f"Initial frequency {cfg.initial_frequency} below min allowed {cfg.min_allowed_frequency}"
            )
        if cfg.benchmark_time / cfg.sample_interval < cfg.min_samples:
            raise ValueError(
                "Benchmark time too short vs sample interval; not enough samples"
            )

    def _fetch_default_settings(self) -> None:
        """
        Get small_core_count, asic_count, and reasonable default voltage/frequency.
        Does NOT hard-fail if smallCoreCount is missing.
        """
        # /api/system/info
        url_info = f"{self.base_url}/api/system/info"
        r = requests.get(url_info, timeout=10)
        r.raise_for_status()
        system_info = r.json()

        self.small_core_count = system_info.get("smallCoreCount")
        self.asic_count = system_info.get("asicCount")

        self.default_voltage = system_info.get("coreVoltage", 1150)
        self.default_frequency = system_info.get("frequency", 500)

        # /api/system/asic (optional but helpful)
        url_asic = f"{self.base_url}/api/system/asic"
        try:
            r2 = requests.get(url_asic, timeout=10)
            r2.raise_for_status()
            asic_info = r2.json()
        except Exception:
            asic_info = {}

        if isinstance(asic_info, dict):
            self.default_voltage = asic_info.get("defaultVoltage", self.default_voltage)
            self.default_frequency = asic_info.get(
                "defaultFrequency", self.default_frequency
            )
            if self.asic_count is None:
                self.asic_count = asic_info.get("asicCount", self.asic_count)

    def _get_system_info(self) -> Optional[Dict[str, Any]]:
        retries = 3
        for _ in range(retries):
            try:
                r = requests.get(f"{self.base_url}/api/system/info", timeout=10)
                r.raise_for_status()
                return r.json()
            except requests.exceptions.RequestException:
                time.sleep(5)
        return None

    def _set_system_settings(self, core_voltage: int, frequency: int) -> None:
        payload = {
            "coreVoltage": core_voltage,
            "frequency": frequency,
        }
        r = requests.patch(f"{self.base_url}/api/system", json=payload, timeout=10)
        r.raise_for_status()
        # restart + wait for stabilization
        self._restart_system(wait=self.config.sleep_time)

    def _restart_system(
        self,
        wait: int = 0,
        poll_timeout: int = 120,
        poll_interval: int = 5,
    ) -> None:
        """Restart the target system, then wait for it to become reachable again.

        Flow:
          1) POST /api/system/restart
          2) Sleep for `wait` seconds (stabilization window)
          3) Poll GET /api/system/info every `poll_interval` seconds for up to
             `poll_timeout` seconds. If still unreachable, raise TimeoutError.
        """
        r = requests.post(f"{self.base_url}/api/system/restart", timeout=10)
        r.raise_for_status()

        if wait > 0:
            time.sleep(wait)

        if poll_timeout <= 0:
            return

        deadline = time.time() + poll_timeout
        last_exc: Optional[Exception] = None

        while time.time() < deadline:
            try:
                r = requests.get(f"{self.base_url}/api/system/info", timeout=10)
                r.raise_for_status()
                return
            except requests.exceptions.RequestException as e:
                last_exc = e
                time.sleep(poll_interval)

        raise TimeoutError(
            f"System at {self.base_url} did not become reachable within "
            f"{poll_timeout}s after restart. Last error: {last_exc}"
        )

    # --- core benchmark iteration ---

    def _benchmark_iteration(
        self, core_voltage: int, frequency: int
    ) -> tuple[
        Optional[float],
        Optional[float],
        Optional[float],
        bool,
        Optional[float],
        Optional[str],
        Optional[Dict[str, Any]],
    ]:
        """
        Returns:
          average_hashrate, average_temp, efficiency_jth, hashrate_ok,
          avg_vr_temp, error_reason, extras
        """
        cfg = self.config
        error_reason_local: Optional[str] = None
        hash_rates: List[float] = []
        temps: List[float] = []
        powers: List[float] = []
        vr_temps: List[float] = []
        fan_pcts: List[float] = []
        fan_rpms: List[float] = []
        error_pcts: List[float] = []

        # Stratum share counters (since boot). We track start/end per step so we can
        # compute rejection rate for the step (delta-based to be robust even if the
        # counters aren't perfectly zero right after a reboot).
        shares_acc_start: Optional[int] = None
        shares_rej_start: Optional[int] = None
        shares_acc_end: Optional[int] = None
        shares_rej_end: Optional[int] = None

        # hashrateMonitor domain tracking:
        # domains_samples[asic_index][domain_index] = [samples...]
        domains_samples: List[List[List[float]]] = []
        # per-ASIC total hashrate samples (hashrateMonitor.asics[*].total)
        asic_total_samples: List[List[float]] = []
        # per-ASIC errorCount start/end
        asic_error_start: List[Optional[int]] = []
        asic_error_end: List[Optional[int]] = []

        extras: Dict[str, Any] = {}

        total_samples = cfg.benchmark_time // cfg.sample_interval

        expected_from_cores: Optional[float] = None
        if self.small_core_count and self.asic_count:
            try:
                expected_from_cores = frequency * (
                    (self.small_core_count * self.asic_count) / 1000.0
                )
            except Exception:
                expected_from_cores = None

        expected_hashrate: Optional[float] = None  # from API if available

        for i in range(total_samples):
            if self._stop.is_set():
                return None, None, None, False, None, "CANCELLED", None

            info = self._get_system_info()
            if info is None:
                return None, None, None, False, None, "SYSTEM_INFO_FAILURE", None

            # Grab firmware's own expectedHashrate if not yet set
            if expected_hashrate is None:
                maybe_exp = info.get("expectedHashrate")
                if isinstance(maybe_exp, (int, float)) and maybe_exp > 0:
                    expected_hashrate = float(maybe_exp)

            temp = info.get("temp")
            vr_temp = info.get("vrTemp")
            voltage = info.get("voltage")
            hash_rate = info.get("hashRate")
            power = info.get("power")
            fanspeed = info.get("fanspeed")          # bitaxe field
            fanrpm = info.get("fanrpm")              # bitaxe field
            error_percentage = info.get("errorPercentage")
            hashrate_monitor = info.get("hashrateMonitor")

            # Shares accepted/rejected (since boot). Prefer top-level keys, but fall
            # back to stratum.pools[0] if needed.
            acc_raw = info.get("sharesAccepted")
            rej_raw = info.get("sharesRejected")
            if acc_raw is None or rej_raw is None:
                try:
                    pools = (info.get("stratum") or {}).get("pools")
                    if isinstance(pools, list) and pools:
                        p0 = pools[0] if isinstance(pools[0], dict) else None
                        if p0:
                            acc_raw = acc_raw if acc_raw is not None else p0.get("accepted")
                            rej_raw = rej_raw if rej_raw is not None else p0.get("rejected")
                except Exception:
                    # shares are optional; ignore any parsing errors
                    pass

            try:
                acc_int = int(acc_raw) if acc_raw is not None else None
            except Exception:
                acc_int = None
            try:
                rej_int = int(rej_raw) if rej_raw is not None else None
            except Exception:
                rej_int = None

            if acc_int is not None and rej_int is not None:
                if shares_acc_start is None:
                    shares_acc_start = acc_int
                if shares_rej_start is None:
                    shares_rej_start = rej_int
                shares_acc_end = acc_int
                shares_rej_end = rej_int

            if temp is None or voltage is None:
                return None, None, None, False, None, "MISSING_TEMP_OR_VOLTAGE", None

            if temp < 5:
                return None, None, None, False, None, "TEMPERATURE_BELOW_5", None

            if temp >= cfg.max_temp:
                return None, None, None, False, None, "CHIP_TEMP_EXCEEDED", None

            if vr_temp is not None and vr_temp >= cfg.max_vr_temp:
                return None, None, None, False, None, "VR_TEMP_EXCEEDED", None

            if voltage < cfg.min_input_voltage:
                return None, None, None, False, None, "INPUT_VOLTAGE_BELOW_MIN", None

            if voltage > cfg.max_input_voltage:
                return None, None, None, False, None, "INPUT_VOLTAGE_ABOVE_MAX", None

            if hash_rate is None or power is None:
                return None, None, None, False, None, "MISSING_HASHRATE_OR_POWER", None

            if power > cfg.max_power:
                return None, None, None, False, None, "POWER_CONSUMPTION_EXCEEDED", None

            hash_rates.append(float(hash_rate))
            temps.append(float(temp))
            powers.append(float(power))
            if vr_temp is not None and vr_temp > 0:
                vr_temps.append(float(vr_temp))
            if isinstance(fanspeed, (int, float)):
                fan_pcts.append(float(fanspeed))
            if isinstance(fanrpm, (int, float)):
                fan_rpms.append(float(fanrpm))
            if isinstance(error_percentage, (int, float)):
                error_pcts.append(float(error_percentage))

            # hashrate registers tracking (Gamma + Nerdqaxe++). Graceful if missing.
            if isinstance(hashrate_monitor, dict):
                asics = hashrate_monitor.get("asics")
                if isinstance(asics, list):
                    for asic_index, asic_entry in enumerate(asics):
                        if not isinstance(asic_entry, dict):
                            continue

                        # ensure per-ASIC arrays exist
                        while len(domains_samples) <= asic_index:
                            domains_samples.append([])
                        while len(asic_total_samples) <= asic_index:
                            asic_total_samples.append([])
                        while len(asic_error_start) <= asic_index:
                            asic_error_start.append(None)
                        while len(asic_error_end) <= asic_index:
                            asic_error_end.append(None)

                        # total hashrate per ASIC (optional but useful)
                        maybe_total = asic_entry.get("total")
                        if isinstance(maybe_total, (int, float)):
                            asic_total_samples[asic_index].append(float(maybe_total))

                        # errorCount per ASIC (we track start/end + delta)
                        maybe_ec = asic_entry.get("errorCount")
                        if isinstance(maybe_ec, (int, float)):
                            ec_int = int(maybe_ec)
                            if asic_error_start[asic_index] is None:
                                asic_error_start[asic_index] = ec_int
                            asic_error_end[asic_index] = ec_int

                        # per-domain hashrate registers
                        domains = asic_entry.get("domains")
                        if isinstance(domains, list):
                            # ensure per-domain lists
                            while len(domains_samples[asic_index]) < len(domains):
                                domains_samples[asic_index].append([])
                            for domain_index, val in enumerate(domains):
                                if isinstance(val, (int, float)):
                                    domains_samples[asic_index][domain_index].append(float(val))

            with self._lock:
                # last_sample is per-iteration; progress is global (updated via _update_eta)
                self.last_sample = {
                    "sample": i + 1,
                    "totalSamples": total_samples,
                    "hashRate": hash_rate,
                    "temp": temp,
                    "vrTemp": vr_temp,
                    "voltage": voltage,
                    "power": power,
                    "fanspeed": fanspeed,
                    "fanrpm": fanrpm,
                    "errorPercentage": error_percentage,
                    "sharesAccepted": acc_int,
                    "sharesRejected": rej_int,
                }
                # update ETA + global progress using current iteration + sample index
                self._update_eta(self._iteration_index, i + 1, total_samples)

            if i < total_samples - 1:
                time.sleep(cfg.sample_interval)

        if not hash_rates or not temps or not powers:
            return None, None, None, False, None, "NO_DATA_COLLECTED", None

        # Trim outliers for hashrate: drop 3 highest + 3 lowest if enough samples
        sorted_hash = sorted(hash_rates)
        trimmed_hash = (
            sorted_hash[3:-3] if len(sorted_hash) > 6 else sorted_hash
        )
        avg_hash = sum(trimmed_hash) / len(trimmed_hash) if trimmed_hash else 0.0

        # Remove first 6 temps (warmup) if possible
        sorted_temp = sorted(temps)
        trimmed_temp = sorted_temp[6:] if len(sorted_temp) > 6 else sorted_temp
        avg_temp = sum(trimmed_temp) / len(trimmed_temp) if trimmed_temp else 0.0

        avg_vr_temp = None
        if vr_temps:
            sorted_vr = sorted(vr_temps)
            trimmed_vr = sorted_vr[6:] if len(sorted_vr) > 6 else sorted_vr
            if trimmed_vr:
                avg_vr_temp = sum(trimmed_vr) / len(trimmed_vr)

        avg_power = sum(powers) / len(powers)
        avg_fan_pct: Optional[float] = None
        if fan_pcts:
            avg_fan_pct = sum(fan_pcts) / len(fan_pcts)
        avg_fan_rpm: Optional[float] = None
        if fan_rpms:
            avg_fan_rpm = sum(fan_rpms) / len(fan_rpms)
        avg_error_pct: Optional[float] = None
        if error_pcts:
            avg_error_pct = sum(error_pcts) / len(error_pcts)

        if avg_hash <= 0:
            return None, None, None, False, avg_vr_temp, "ZERO_HASHRATE", None

        efficiency_jth = avg_power / (avg_hash / 1000.0)

        # choose best guess for expected hashrate:
        # prefer firmware's expectedHashrate, fallback to core-count math
        if expected_hashrate is None:
            expected_hashrate = expected_from_cores

        if expected_hashrate is not None and expected_hashrate > 0:
            mode = (self.config.sweep_mode or "").lower()
            # Default for adaptive runs: 94% of expected
            target_ratio = 0.94
            # Looser threshold for grid testing
            if mode == "grid":
                target_ratio = 0.80

            hashrate_ok = (avg_hash >= expected_hashrate * target_ratio)
        else:
            # no idea what "expected" is; don't penalize
            hashrate_ok = True
            
        # --- Fold ASIC error % into "is this point good?" (configurable) ---
        threshold = self.config.error_rate_warn_threshold
        if threshold is None:
            threshold = 2.0  # default if not set in config / UI

        if avg_error_pct is not None and avg_error_pct > threshold:
            # Treat this point as "not good enough", which will cause the
            # outer loop to raise V / back off F and re-test.
            hashrate_ok = False
            # IMPORTANT: do NOT set error_reason_local here; this is not a hard stop
            #error_reason_local = "ERROR_RATE_TOO_HIGH"

        # --- build hashrate domain statistics (min/max/avg/stddev per domain) ---
        hashrate_domains_stats: List[Dict[str, Any]] = []
        for asic_index, asic_domains in enumerate(domains_samples):
            if not isinstance(asic_domains, list):
                continue

            domain_stats_list: List[Dict[str, Any]] = []
            for domain_index, samples in enumerate(asic_domains):
                if not samples:
                    continue
                mn = min(samples)
                mx = max(samples)
                avg = sum(samples) / len(samples)
                # population stddev
                var = sum((x - avg) ** 2 for x in samples) / len(samples)
                stddev = var ** 0.5

                domain_stats_list.append(
                    {
                        "index": domain_index,
                        "min": mn,
                        "max": mx,
                        "avg": avg,
                        "stddev": stddev,
                    }
                )

            if not domain_stats_list and not (asic_total_samples and asic_total_samples[asic_index]) \
               and (asic_error_start[asic_index] is None and asic_error_end[asic_index] is None):
                continue

            asic_entry: Dict[str, Any] = {
                "asicIndex": asic_index,
                "domains": domain_stats_list,
            }

            if asic_index < len(asic_total_samples) and asic_total_samples[asic_index]:
                totals = asic_total_samples[asic_index]
                asic_entry["avgTotal"] = sum(totals) / len(totals)

            if asic_index < len(asic_error_start):
                ec_start = asic_error_start[asic_index]
                ec_end = asic_error_end[asic_index]
                if ec_start is not None:
                    asic_entry["errorCountStart"] = ec_start
                if ec_end is not None:
                    asic_entry["errorCountEnd"] = ec_end
                    if ec_start is not None:
                        asic_entry["errorCountDelta"] = ec_end - ec_start

            hashrate_domains_stats.append(asic_entry)

        # populate extras dict
        extras["avgPower"] = avg_power
        if avg_fan_pct is not None:
            extras["avgFanPct"] = avg_fan_pct
        if avg_fan_rpm is not None:
            extras["avgFanRpm"] = avg_fan_rpm
        if avg_error_pct is not None:
            extras["avgErrorPercentage"] = avg_error_pct
        if hashrate_domains_stats:
            extras["hashrateDomains"] = hashrate_domains_stats
        # Also record what the expected hashrate was for this point (if known)
        if expected_hashrate is not None:
            extras["expectedHashrate"] = expected_hashrate

        # Shares + reject rate (delta within this step)
        if (
            shares_acc_start is not None
            and shares_rej_start is not None
            and shares_acc_end is not None
            and shares_rej_end is not None
        ):
            acc_delta = max(0, shares_acc_end - shares_acc_start)
            rej_delta = max(0, shares_rej_end - shares_rej_start)
            total_delta = acc_delta + rej_delta

            extras["sharesAcceptedStart"] = shares_acc_start
            extras["sharesRejectedStart"] = shares_rej_start
            extras["sharesAcceptedEnd"] = shares_acc_end
            extras["sharesRejectedEnd"] = shares_rej_end
            extras["sharesAccepted"] = acc_delta
            extras["sharesRejected"] = rej_delta
            extras["sharesTotal"] = total_delta

            if total_delta > 0:
                extras["rejectRate"] = (rej_delta / total_delta) * 100.0
        # Optional: record what threshold was used for this run
        extras["errorRateWarnThreshold"] = threshold

        return avg_hash, avg_temp, efficiency_jth, hashrate_ok, avg_vr_temp, error_reason_local, (extras or None)

    # --- cleanup & results writing ---

    def _reset_to_best_or_default(self) -> None:
        if self.default_voltage is None or self.default_frequency is None:
            return

        if not self.results:
            self._set_system_settings(self.default_voltage, self.default_frequency)
            return

        global_threshold = self.config.error_rate_warn_threshold
        if global_threshold is None:
            global_threshold = 2.0

        def is_acceptable(r: Dict[str, Any]) -> bool:
            # Skip error-only / non-numeric rows
            if not isinstance(r, dict):
                return False
            if "averageHashRate" not in r or "efficiencyJTH" not in r:
                return False
            if r.get("valid") is False:
                return False

            err = r.get("errorPercentage")
            thr = r.get("errorRateWarnThreshold", global_threshold)

            if err is None or thr is None:
                return True
            return err <= thr

        candidates = [r for r in self.results if is_acceptable(r)]

        if not candidates:
            self._set_system_settings(self.default_voltage, self.default_frequency)
            return

        best = max(candidates, key=lambda x: x["averageHashRate"])
        v = best["coreVoltage"]
        f = best["frequency"]
        self._set_system_settings(v, f)


    def _write_results_json(self) -> str:
        start = self.start_time or datetime.utcnow().isoformat()
        start_ts = start.replace(":", "").replace("-", "")
        filename = f"run_{self.run_id}_{self.bitaxe_ip_raw.replace('.', '-')}_{start_ts}.json"

        base_dir = os.path.join("data", "results")
        os.makedirs(base_dir, exist_ok=True)
        path = os.path.join(base_dir, filename)

        # Only numeric rows for "top" lists
        numeric_results = [
            r for r in self.results
            if isinstance(r, dict)
            and "averageHashRate" in r
            and "efficiencyJTH" in r
        ]

        if numeric_results:
            top_by_hash = sorted(
                numeric_results, key=lambda x: x["averageHashRate"], reverse=True
            )[:5]
            top_by_eff = sorted(
                numeric_results, key=lambda x: x["efficiencyJTH"]
            )[:5]
        else:
            top_by_hash = []
            top_by_eff = []

        final_data = {
            "runId": self.run_id,
            "bitaxeIp": self.bitaxe_ip_raw,
            "deviceType": self.config.device_type.value,
            "startTime": self.start_time,
            "endTime": self.end_time,
            "config": asdict(self.config),
            # Keep *all* rows, including error-only ones
            "results": self.results,
            "topPerformers": top_by_hash,
            "mostEfficient": top_by_eff,
        }

        with open(path, "w") as f:
            json.dump(final_data, f, indent=2)

        return path


class GridBenchmarkRunner(BenchmarkRunner):
    """
    Variant of BenchmarkRunner that does a full grid sweep across
    [min_allowed_voltage..max_allowed_voltage] and
    [min_allowed_frequency..max_allowed_frequency] using the configured increments.
    """

    def _estimate_total_iterations(self) -> int:
        cfg = self.config
        if cfg.frequency_increment <= 0 or cfg.voltage_increment <= 0:
            return 1

        v_min = min(cfg.min_allowed_voltage, cfg.max_allowed_voltage)
        v_max = max(cfg.min_allowed_voltage, cfg.max_allowed_voltage)
        f_min = min(cfg.min_allowed_frequency, cfg.max_allowed_frequency)
        f_max = max(cfg.min_allowed_frequency, cfg.max_allowed_frequency)

        vol_steps = max(
            1, (v_max - v_min) // cfg.voltage_increment + 1
        )
        freq_steps = max(
            1, (f_max - f_min) // cfg.frequency_increment + 1
        )
        return vol_steps * freq_steps

    def _build_status_detail_grid(
        self,
        iteration_index: int,
        total_iters: int,
        current_voltage: int,
        current_frequency: int,
    ) -> str:
        iter_label = f"grid point {iteration_index + 1} of ~{total_iters}"
        return (
            f"Grid sweep: testing {current_frequency} MHz @ {current_voltage} mV "
            f"({iter_label}). Will stop early if thermal, power, or input voltage "
            f"limits are exceeded."
        )

    def _run(self) -> None:
        self.start_time = datetime.utcnow().isoformat()
        self._start_dt = datetime.utcnow()
        self.status = BenchmarkStatus.RUNNING
        result_path: Optional[str] = None

        try:
            self._validate_config()
            self._fetch_default_settings()

            self._approx_total_iterations = self._estimate_total_iterations()
            cfg = self.config

            v_min = min(cfg.min_allowed_voltage, cfg.max_allowed_voltage)
            v_max = max(cfg.min_allowed_voltage, cfg.max_allowed_voltage)
            f_min = min(cfg.min_allowed_frequency, cfg.max_allowed_frequency)
            f_max = max(cfg.min_allowed_frequency, cfg.max_allowed_frequency)

            voltages = list(range(v_min, v_max + 1, cfg.voltage_increment))
            freqs = list(range(f_min, f_max + 1, cfg.frequency_increment))

            iteration_index = 0
            stop_all = False
            
            THERMAL_OR_POWER_LIMIT_ERRORS = {
                "CHIP_TEMP_EXCEEDED",
                "VR_TEMP_EXCEEDED",
                "POWER_CONSUMPTION_EXCEEDED",
                "INPUT_VOLTAGE_BELOW_MIN",
                "INPUT_VOLTAGE_ABOVE_MAX",
                "ERROR_RATE_TOO_HIGH",
                "ZERO_HASHRATE",
            }

            FATAL_ERRORS = {
                "SYSTEM_INFO_FAILURE",
                "MISSING_TEMP_OR_VOLTAGE",
                "TEMPERATURE_BELOW_5",
                "MISSING_HASHRATE_OR_POWER",
                "NO_DATA_COLLECTED",
                # add anything else you consider “test is now meaningless”
            }

            for v in voltages:
                if stop_all:
                    break
                stop_after_this_point = False
                for fi, f in enumerate(freqs):
                    if self._stop.is_set():
                        self.status = BenchmarkStatus.CANCELLED
                        self.error_reason = "Cancelled by user"
                        with self._lock:
                            self.status_detail = (
                                "Benchmark cancelled by user; cleaning up and "
                                "restoring device settings."
                            )
                        stop_all = True
                        break

                    self.current_voltage = v
                    self.current_frequency = f

                    with self._lock:
                        self.status_detail = self._build_status_detail_grid(
                            iteration_index,
                            self._approx_total_iterations,
                            v,
                            f,
                        )

                    self._set_system_settings(v, f)
                    self._iteration_index = iteration_index

                    (
                        avg_hashrate,
                        avg_temp,
                        efficiency_jth,
                        hashrate_ok,   # unused for stepping; we still record it
                        avg_vr_temp,
                        error_reason,
                        extras,
                    ) = self._benchmark_iteration(v, f)

                    iteration_index += 1

                    if self._stop.is_set():
                        self.status = BenchmarkStatus.CANCELLED
                        self.error_reason = "Cancelled by user"
                        with self._lock:
                            self.status_detail = (
                                "Benchmark cancelled by user; cleaning up and "
                                "restoring device settings."
                            )
                        stop_all = True
                        break

                    if error_reason is not None:
                        # hit thermal / power / voltage limit – treat as a ceiling,
                        # but still keep any points we collected so far.
                        self.error_reason = error_reason
                        if error_reason in THERMAL_OR_POWER_LIMIT_ERRORS:
                            remaining_at_this_voltage = len(freqs) - (fi + 1)
                            if remaining_at_this_voltage > 0:
                                # shrink our estimate so ETA stops counting skipped points
                                self._approx_total_iterations = max(
                                    iteration_index,  # never less than what we've already done
                                    self._approx_total_iterations - remaining_at_this_voltage,
                                )
                            # Record the failing grid point
                            self.results.append({
                                "runId": self.run_id,
                                "device": self.config.device_type.value,
                                "coreVoltage": v,
                                "frequency": f,
                                "errorReason": error_reason,
                                "valid": False,
                            })

                            with self._lock:
                                self.status_detail = (
                                    f"Grid: {self._describe_error_reason(error_reason)} at "
                                    f"{f} MHz @ {v} mV — stopping higher frequencies at this voltage."
                                )

                            # Early-exit for this *voltage*: do not test higher freqs at this V
                            break

                        # Everything else is treated as a fatal problem; bail out
                        if error_reason in FATAL_ERRORS:
                            # we’re done with the sweep entirely; adjust total iterations to what we’ve done
                            self._approx_total_iterations = iteration_index
                            with self._lock:
                                self.status_detail = (
                                    "Stopping grid sweep: "
                                    f"{self._describe_error_reason(error_reason)}."
                                )
                            stop_all = True
                            break
                        if not self.results:
                            self.status = BenchmarkStatus.FAILED
                        with self._lock:
                            self.status_detail = (
                                f"Stopping grid sweep: "
                                f"{self._describe_error_reason(error_reason)}."
                            )
                        stop_all = True
                        break
                    
                    # If we didn't hit a hard error, but hashrate_ok is False, treat this as a
                    # "soft ceiling" for this voltage: record the point, then stop stepping higher F.
                    if error_reason is None and hashrate_ok is False:
                        # We still want to record the point we just measured
                        # (that happens in the block below), so don't `continue` or `return`.

                        remaining_at_this_voltage = len(freqs) - (fi + 1)
                        if remaining_at_this_voltage > 0:
                            # Shrink ETA so we don't count skipped points at this V
                            self._approx_total_iterations = max(
                                iteration_index,
                                self._approx_total_iterations - remaining_at_this_voltage,
                            )

                        with self._lock:
                            self.status_detail = (
                                "Grid: hashrate under expected at "
                                f"{f} MHz @ {v} mV — stopping higher frequencies at this voltage."
                            )

                        # We'll break out of the freq loop *after* we append this result below.
                        stop_after_this_point = True
                    else:
                        stop_after_this_point = False

                    if (
                        avg_hashrate is not None
                        and avg_temp is not None
                        and efficiency_jth is not None
                    ):
                        avg_power = extras.get("avgPower") if extras else None
                        avg_fan_pct = extras.get("avgFanPct") if extras else None
                        avg_fan_rpm = extras.get("avgFanRpm") if extras else None
                        hashrate_domains = extras.get("hashrateDomains") if extras else None
                        avg_err_pct = extras.get("avgErrorPercentage") if extras else None
                        expected_hash = extras.get("expectedHashrate") if extras else None
                        used_threshold = extras.get("errorRateWarnThreshold")
                        reject_rate = extras.get("rejectRate") if extras else None
                        shares_acc = extras.get("sharesAccepted") if extras else None
                        shares_rej = extras.get("sharesRejected") if extras else None
                        shares_total = extras.get("sharesTotal") if extras else None

                        result: Dict[str, Any] = {
                            "coreVoltage": v,
                            "frequency": f,
                            "averageHashRate": avg_hashrate,
                            "averageTemperature": avg_temp,
                            "efficiencyJTH": efficiency_jth,
                        }
                        if avg_power is not None:
                            result["averagePower"] = avg_power
                        if avg_vr_temp is not None:
                            result["averageVRTemp"] = avg_vr_temp
                        if avg_fan_pct is not None:
                            result["fanSpeed"] = avg_fan_pct
                        if avg_fan_rpm is not None:
                            result["fanRPM"] = avg_fan_rpm
                        if avg_err_pct is not None:
                            result["errorPercentage"] = avg_err_pct
                        if reject_rate is not None:
                            result["rejectRate"] = reject_rate
                        if shares_acc is not None:
                            result["sharesAccepted"] = shares_acc
                        if shares_rej is not None:
                            result["sharesRejected"] = shares_rej
                        if shares_total is not None:
                            result["sharesTotal"] = shares_total
                        if hashrate_domains is not None:
                            result["hashrateDomains"] = hashrate_domains
                        if used_threshold is not None:
                            result["errorRateWarnThreshold"] = used_threshold
                        if expected_hash is not None:
                            result["expectedHashrate"] = expected_hash

                        self.results.append(result)

                        # Update "best" metrics same as adaptive mode
                        if self.best_hashrate is None or avg_hashrate > self.best_hashrate:
                            self.best_hashrate = avg_hashrate
                        if self.best_efficiency is None or efficiency_jth < self.best_efficiency:
                            self.best_efficiency = efficiency_jth
                        
                        if stop_after_this_point:
                            break  # stop stepping to higher frequencies at this V
    
                    else:
                        with self._lock:
                            self.status_detail = (
                                "Benchmark stopping: no valid data collected at this grid point."
                            )
                        stop_all = True
                        break

            # After grid completes (or stops early), restore device
            self._reset_to_best_or_default()
            if self.results:
                result_path = self._write_results_json()
                if self.status == BenchmarkStatus.RUNNING:
                    self.status = BenchmarkStatus.COMPLETED
                    with self._lock:
                        if not self.status_detail:
                            self.status_detail = (
                                "Grid sweep completed; restored device to "
                                "best-performing settings."
                            )
            else:
                if self.status != BenchmarkStatus.CANCELLED:
                    self.status = BenchmarkStatus.FAILED
                    if self.error_reason is None:
                        self.error_reason = "NO_DATA_COLLECTED"
                    with self._lock:
                        self.status_detail = (
                            f"Benchmark failed: "
                            f"{self._describe_error_reason(self.error_reason)}."
                        )

        except Exception as e:
            self.status = BenchmarkStatus.FAILED
            self.error_reason = f"Unexpected error: {e}"
            with self._lock:
                self.status_detail = f"Benchmark failed due to unexpected error: {e}."
            try:
                self._reset_to_best_or_default()
            except Exception:
                pass
        finally:
            self.end_time = datetime.utcnow().isoformat()
            if self.on_finish:
                try:
                    self.on_finish(self, result_path)
                except Exception:
                    pass


def config_from_dict(data: Dict[str, Any]) -> BenchmarkConfig:
    """
    Helper to construct BenchmarkConfig from arbitrary dict (e.g. from API).
    """
    field_names = {f.name for f in fields(BenchmarkConfig)}
    filtered = {k: v for k, v in data.items() if k in field_names}
    # convert device_type if needed
    dt = filtered.get("device_type") or filtered.get("deviceType")
    if not isinstance(dt, DeviceType):
        filtered["device_type"] = DeviceType(dt)
    return BenchmarkConfig(**filtered)