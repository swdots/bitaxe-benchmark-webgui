# Bitaxe Hashrate Benchmark - Web GUI

Based on mrv777's Python-based benchmarking tool, this is a dockerized web gui for running tests and reviewing results.

## Features

- Automated benchmarking of different voltage/frequency combinations
- Temperature monitoring and safety cutoffs
- Power efficiency calculations (J/TH)
- Automatic saving of benchmark results
- Graceful shutdown with best settings retention

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
      - "8020:8000"          # host:container
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

## Disclaimer

Please use this tool responsibly. Overclocking and voltage modifications can potentially damage your hardware if not done carefully. Always ensure proper cooling and monitor your device during benchmarking.
