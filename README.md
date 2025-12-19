# Bitaxe Hashrate Benchmark - Web GUI

This is a dockerized web gui for running tests and reviewing results. I did use ChatGPT for assistance on this but have also worked on double-checking and tweaking things, so please let me know if there are any issues. Use at your own discretion.

---

![Multiple systems can test at once](multi_test_progress.png)


## Overview

Based on mrv777's Python-based benchmarking tool, this project is a web-based benchmarking harness for Bitaxe / NerdQaxe-class devices.
It discovers your device when given an IP address, applies a (hopefully) safe but configurable tuning profile, then walks through voltage/frequency steps while collecting detailed telemetry and generating results you can compare, export, and re-run later.

---

## Key Features

### Device detection & profiles

* **Automatic device identification**

  * Queries `/api/system/info` and `/api/system/asic` to detect:

    * Device model / ASIC model
    * Board revision
    * ASIC count & small core count
    * Firmware & AxeOS versions
  * Classifies into:

    * `gamma_602` (Bitaxe Gamma 602)
    * `nerdqaxe_pp` (NerdQaxe++)
    * `other` (generic / unknown)
* **Built-in profiles**

  * Safe default limits for each device type:

    * Max chip temperature
    * Max VR temperature
    * Max power
    * Voltage and frequency bounds
* **Custom profiles**

  * Full configuration is editable in the UI:

    * Initial core voltage / frequency
    * Voltage & frequency increments
    * Temperature and power limits
    * Input voltage bounds
    * Benchmark duration, sample interval, min samples, sleep time
  * Save any configuration as a named custom profile.
  * Profiles can be listed, selected, and deleted via the UI/API.

---

### Benchmark engine

* **Automated search over V/F space**

  * Starts from a configurable initial `(voltage, frequency)` pair.
  * Iteratively:

    * Applies new settings via `/api/system`.
    * Restarts the device and waits for stabilization.
    * Samples telemetry from `/api/system/info`.
  * Sampling loop:

    * Takes periodic samples (e.g. every 15s) for a fixed benchmark time.
    * Validates readings and enforces safety checks:

      * Max chip temp
      * Max VR temp
      * Min/max input voltage
      * Max power
      * Non-zero hashrate & power
* **Hashrate quality & decision logic**

  * Computes:

    * Average hashrate (GH/s) with basic outlier trimming.
    * Average chip temperature, VR temperature, and power.
    * Efficiency (J/TH).
  * Estimates expected hashrate from:

    * Firmware’s `expectedHashrate` (preferred), or
    * Core/ASIC count + frequency.
  * Decides next step:

    * If hashrate is good:

      * Increase frequency by `frequency_increment`, up to `max_allowed_frequency`.
    * If hashrate is low:

      * Increase voltage by `voltage_increment` and back off frequency.
    * Stops when limits are reached or safety constraints are hit.
* **Best-point restoration**

  * After the run:

    * Picks the best-performing point (highest GH/s) if available.
    * Falls back to device defaults if no valid data.
    * Restores the device to that state before exiting.

---

### Safety, robustness & progress tracking

* **Safety-first guardrails**

  * Hard bounds on:

    * Core voltage (`min_allowed_voltage` / `max_allowed_voltage`)
    * Frequency (`min_allowed_frequency` / `max_allowed_frequency`)
    * Temperature & VR temperature
    * Power & input voltage
  * Sampling abort reasons are explicitly coded (e.g. `CHIP_TEMP_EXCEEDED`, `POWER_CONSUMPTION_EXCEEDED`, `ZERO_HASHRATE`, etc.).
* **Global progress & ETA**

  * Estimates total number of iterations from configured voltage/frequency ranges and increments.
  * Tracks progress within each iteration and converts into:

    * **Global progress %** for the entire run.
    * **ETA (seconds/minutes)** based on elapsed time and completion fraction.
* **Cancellation**

  * Benchmarks run in background threads.
  * Runs can be cancelled via API/UI.
  * On cancel, the runner:

    * Stops sampling gracefully.
    * Restores device settings.
    * Marks the run as cancelled.

---

### Narrative status & live insights

* **Live “statusDetail” narrative**

  * For each step, the backend builds a human-friendly explanation:

    * What it’s doing:

      * *“Testing 525 MHz @ 1150 mV (step 1 of ~N)”*
    * Why it chose that step:

      * *“Previous run passed thermal and hashrate checks; increasing frequency by +25 MHz.”*
    * What’s next:

      * *“Next target 550 MHz @ 1150 mV if stable.”*
    * Includes context from the previous step:

      * *“(last point: 1511.3 GH/s, 60.0 °C, 14.30 J/TH)”*
* **Recent point history**

  * Each active run exposes the last 10 completed test points:

    * Core voltage
    * Frequency
    * Average hashrate (GH/s)
    * Average chip & VR temperature
    * Efficiency (J/TH)
  * Visible under each running benchmark as a toggleable mini-table, so you don’t have to wait until the run finishes to see how it’s going.

---

### Web UI

* **Single-page control panel**

  * Built with plain HTML/CSS/JS (served by FastAPI).
  * Sections:

    * **Start new benchmark**

      * Target IP input.
      * Device type and profile selector.
      * Auto-detect button to pull device info + recommended config.
      * Editable configuration grid for all tuning parameters.
      * Optional profile save.
      * Run notes input (“after repaste”, fan curve changes, etc.).
    * **Connected device info**

      * Shows:

        * Model, ASIC model, board revision
        * ASIC count and small core count
        * Firmware / AxeOS versions
        * Hostname
      * Current runtime snapshot:

        * Voltage & frequency
        * Temperature & power
        * Hashrate & expected hashrate
        * Fan %, RPM, temperature target
    * **Running benchmarks**

      * Lists all active runs with:

        * Run ID, device type, IP.
        * Live best GH/s and J/TH so far.
        * Global progress bar (%).
        * ETA in minutes.
        * Narrative `statusDetail` line explaining what’s happening and what’s next.
        * Click-to-toggle mini table showing last 10 completed points.
        * Cancel button.
    * **Past benchmarks**

      * History table with:

        * Run ID, device, IP
        * Status (pending, running, completed, failed, cancelled)
        * Best GH/s and best J/TH
        * Start & finish timestamps
      * Actions per run:

        * **Load config** → populates the “Start new benchmark” form with that run’s config.
        * **Details** → opens a details view for that run.
        * **Delete** → removes completed/failed/cancelled runs and associated result files.
    * **Run details**

      * Summary of:

        * Best hashrate point.
        * Most efficient point.
      * Full results JSON (currently), with planned structured tables & charts.
      * Notes display, with API support to update run notes.

---

### Notes & annotations

* **Per-run notes**

  * Each benchmark run stores a free-form notes field:

    * e.g. “After repasting with Kryonaut”, “Stock fan, 25 °C ambient”, “Custom shroud v2”.
  * Notes can be set at run creation from the UI.
  * Notes can be edited later via a dedicated API endpoint and the details view.

---

### Data storage & exports

* **Database**

  * Uses SQLite to store:

    * Run metadata (status, device, IP, start/end times).
    * Profile references and serialized configs.
    * Best hashrate and efficiency summary.
    * Path to the results JSON file.
    * Notes.
* **Result files**

  * Each completed run writes a JSON file under `data/results/` with:

    * Full list of measured points (`results`).
    * Derived “top performers” list (highest hashrate).
    * Derived “most efficient” list (lowest J/TH).
* **Deletion**

  * Completed, failed, and cancelled runs can be deleted:

    * Via `DELETE /api/benchmarks/{run_id}`.
    * Automatically removes the associated results file (best effort).

---

### API highlights

* **Device identification**

  * `POST /api/devices/identify` → returns device type, recommended config, metadata, and starting readings.
* **Profiles**

  * `GET /api/profiles` → list built-in + custom profiles.
  * `POST /api/profiles` → create a custom profile.
  * `DELETE /api/profiles/{profile_id}` → delete a custom profile.
* **Benchmarks**

  * `POST /api/benchmarks` → start a new benchmark run.

    * Rejects if a run is already active for that target IP (prevents duplicate stress on the same device).
  * `GET /api/benchmarks` → list all runs (with live in-memory updates for active ones).
  * `GET /api/benchmarks/{run_id}` → get run details (optionally including results).
  * `POST /api/benchmarks/{run_id}/cancel` → request cancellation.
  * `DELETE /api/benchmarks/{run_id}` → delete a completed/failed/cancelled run.
  * `POST /api/benchmarks/{run_id}/notes` → update notes for a run.

---

### Details

* **Rich details view**

  * Nicely formatted table of all test points (like the mini table, but full run).
  * Highlight:

    * Best hashrate row (orange outline).
    * Best efficiency row (green outline).
* **Charts**

  * Hashrate vs frequency.
  * Efficiency vs frequency.
  * Power vs frequency.
  * Step-by-step “timeline” plot vs test step index:

    * Frequency
    * Power
    * Core voltage
    * Hashrate
    * Efficiency
* **CSV export**

  * One-click CSV download of run data from the details page.
* **Inline notes editing**

  * Rich notes editor directly on the run details page.

## Installation

### Standard Installation

1. Clone the repository

2. Run build.sh to create the docker container image

### Docker compose example:
```version: "3.9"

services:
  bitaxe-bench-web:
    image: bitaxe-bench-web:latest
    container_name: bitaxe-bench-web
    restart: unless-stopped
    ports:
      - "8000:8000"          # host:container (use whatever port you'd like)
    volumes:
      - bitaxe_bench_data:/app/data
    environment:
      # optional, just explicit
      - PYTHONUNBUFFERED=1

volumes:
  bitaxe_bench_data:
    driver: local
```
## Configuration

The script includes several configurable parameters:

- Maximum chip temperature: 66°C
- Maximum VR temperature: 86°C
- Maximum allowed voltage: 1400mV
- Minimum allowed voltage: 1000mV
- Maximum allowed frequency: 1200MHz
- Maximum power consumption: 40W
- Minimum allowed frequency: 400MHz
- Minimum input voltage: 4800mV
- Maximum input voltage: 5500mV
- Benchmark duration: 10 minutes
- Sample interval: 15 seconds
- Sleep time before benchmark: 90 seconds
- **Minimum required samples: 7** (for valid data processing)
- Voltage increment: 20mV
- Frequency increment: 25MHz

## Safety Features

- Automatic temperature monitoring with safety cutoff (66°C chip temp)
- Voltage regulator (VR) temperature monitoring with safety cutoff (86°C)
- Input voltage monitoring with minimum threshold (4800mV) and maximum threshold (5500mV)
- Power consumption monitoring with safety cutoff (40W)
- Temperature validation (must be above 5°C)
- Graceful shutdown on interruption (Ctrl+C)
- Automatic reset to best performing settings after benchmarking
- Input validation for safe voltage and frequency ranges
- Hashrate validation to ensure stability
- Protection against invalid system data
- Outlier removal from benchmark results

## Benchmarking Process

The tool follows this process:
1. Starts with user-specified or default voltage/frequency
2. Tests each combination for 20 minutes
3. Validates hashrate is within 8% of theoretical maximum
4. Incrementally adjusts settings:
   - Increases frequency if stable
   - Increases voltage if unstable
   - Stops at thermal or stability limits
5. Records and ranks all successful configurations
6. Automatically applies the best performing stable settings
7. Restarts system after each test for stability
8. Allows 90-second stabilization period between tests

## Data Processing

The tool implements several data processing techniques to ensure accurate results:
- Removes 3 highest and 3 lowest hashrate readings to eliminate outliers
- Excludes first 6 temperature readings during warmup period
- Validates hashrate is within 6% of theoretical maximum
- Averages power consumption across entire test period
- Monitors VR temperature when available
- Calculates efficiency in Joules per Terahash (J/TH)

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

### Tips ☕
If you like this tool or find it useful and want to buy me a coffee, tips are much appreciated!

- **BTC:** `bc1qa2m3ggt0mygtyzslvs2a96df7kx2te3nqv2szp`
- **DOGE:** `DJsZUqUYEBP99MvpCVxxZKqEdV6GqBWpEs`

## Disclaimer

Please use this tool responsibly. Overclocking and voltage modifications can potentially damage your hardware if not done carefully. Always ensure proper cooling and monitor your device during benchmarking.
