#!/usr/bin/env bash
# ML Heat Diagnostic Script
# Collects system temperature, CPU, GPU, and thermal throttling information
# Usage: sudo ./ml_heat_diag.sh

set -euo pipefail

# Generate timestamp for output file
TS="$(date +%Y%m%d_%H%M%S)"
OUT="$HOME/ml_heat_diag_${TS}.txt"
ARTIFACTS_DIR="$HOME/ml_heat_diag_${TS}_artifacts"
mkdir -p "$ARTIFACTS_DIR"

echo "=== ML HEAT DIAGNOSTIC $(date) ===" > "$OUT"
echo "Output file: $OUT" | tee -a "$OUT"
echo "Artifacts directory: $ARTIFACTS_DIR" | tee -a "$OUT"
echo "" >> "$OUT"

# Check and install required tools
echo "Checking required diagnostic tools..." | tee -a "$OUT"
MISSING_TOOLS=()

command -v sensors >/dev/null 2>&1 || MISSING_TOOLS+=("lm-sensors")
command -v cpupower >/dev/null 2>&1 || MISSING_TOOLS+=("cpupower")
command -v upower >/dev/null 2>&1 || MISSING_TOOLS+=("upower")
command -v smartctl >/dev/null 2>&1 || MISSING_TOOLS+=("smartmontools")
command -v powertop >/dev/null 2>&1 || MISSING_TOOLS+=("powertop")

if [ ${#MISSING_TOOLS[@]} -gt 0 ]; then
    echo "WARNING: Missing tools: ${MISSING_TOOLS[*]}" | tee -a "$OUT"
    echo "Install with: sudo apt install ${MISSING_TOOLS[*]} -y" | tee -a "$OUT"
    echo "" >> "$OUT"
fi

# CPU Temperature and Sensors
echo "=== CPU Temperature (sensors) ===" >> "$OUT"
if command -v sensors >/dev/null 2>&1; then
    sensors >> "$OUT" 2>&1 || echo "No lm-sensors output available" >> "$OUT"
else
    echo "lm-sensors not installed. Install with: sudo apt install lm-sensors -y" >> "$OUT"
fi
echo "" >> "$OUT"

# GPU Temperature (NVIDIA)
echo "=== GPU Temperature (NVIDIA) ===" >> "$OUT"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,utilization.memory,power.draw --format=csv,noheader >> "$OUT" 2>&1 || nvidia-smi >> "$OUT" 2>&1
    echo "" >> "$OUT"
    nvidia-smi >> "$ARTIFACTS_DIR/nvidia-smi_full.txt" 2>&1 || true
else
    echo "NVIDIA drivers not detected (nvidia-smi not found)" >> "$OUT"
fi
echo "" >> "$OUT"

# CPU Frequency Information
echo "=== CPU Frequency Summary ===" >> "$OUT"
if [ -f /proc/cpuinfo ]; then
    echo "Current CPU frequencies:" >> "$OUT"
    grep 'cpu MHz' /proc/cpuinfo | head -8 >> "$OUT" || echo "Could not read CPU frequencies" >> "$OUT"
    echo "" >> "$OUT"
    
    # Get CPU governor info
    if command -v cpupower >/dev/null 2>&1; then
        echo "CPU frequency governors:" >> "$OUT"
        cpupower frequency-info >> "$OUT" 2>&1 || echo "Could not read CPU frequency info" >> "$OUT"
    fi
else
    echo "Could not access /proc/cpuinfo" >> "$OUT"
fi
echo "" >> "$OUT"

# CPU Load and Usage
echo "=== CPU Load ===" >> "$OUT"
uptime >> "$OUT" 2>&1
echo "" >> "$OUT"
if command -v top >/dev/null 2>&1; then
    top -bn1 | head -20 >> "$ARTIFACTS_DIR/top_snapshot.txt" 2>&1 || true
fi
echo "" >> "$OUT"

# Memory Usage
echo "=== Memory Usage ===" >> "$OUT"
free -h >> "$OUT" 2>&1
echo "" >> "$OUT"

# Thermal Throttling Logs
echo "=== Kernel Thermal/Throttling Events (last hour) ===" >> "$OUT"
if command -v journalctl >/dev/null 2>&1; then
    sudo journalctl -k --since '-1 hours' 2>/dev/null | grep -iE 'thermal|throttl|overheat|cpu.*temperature|temperature.*limit' >> "$OUT" || echo "No thermal throttling events in last hour" >> "$OUT"
else
    echo "journalctl not available" >> "$OUT"
fi
echo "" >> "$OUT"

# Check for thermal zones
echo "=== Thermal Zones ===" >> "$OUT"
if [ -d /sys/class/thermal ]; then
    for zone in /sys/class/thermal/thermal_zone*/temp; do
        if [ -f "$zone" ]; then
            temp=$(cat "$zone" 2>/dev/null || echo "N/A")
            zone_num=$(echo "$zone" | grep -oP 'thermal_zone\K[0-9]+')
            type_file="${zone%/*}/type"
            if [ -f "$type_file" ]; then
                zone_type=$(cat "$type_file" 2>/dev/null || echo "unknown")
                # Convert millidegrees to degrees Celsius
                if [[ "$temp" =~ ^[0-9]+$ ]]; then
                    temp_c=$((temp / 1000))
                    echo "Thermal zone $zone_num ($zone_type): ${temp_c}°C" >> "$OUT"
                else
                    echo "Thermal zone $zone_num ($zone_type): $temp" >> "$OUT"
                fi
            fi
        fi
    done
else
    echo "No thermal zones found" >> "$OUT"
fi
echo "" >> "$OUT"

# Turbo Boost Status
echo "=== Turbo Boost Status ===" >> "$OUT"
if [ -f /sys/devices/system/cpu/intel_pstate/no_turbo ]; then
    no_turbo=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || echo "N/A")
    if [ "$no_turbo" = "0" ]; then
        echo "Turbo Boost: ENABLED" >> "$OUT"
    elif [ "$no_turbo" = "1" ]; then
        echo "Turbo Boost: DISABLED" >> "$OUT"
    else
        echo "Turbo Boost: Status unknown ($no_turbo)" >> "$OUT"
    fi
else
    echo "Intel P-State not available (non-Intel CPU or different scaling driver)" >> "$OUT"
fi
echo "" >> "$OUT"

# Power Profile
echo "=== Power Profile ===" >> "$OUT"
if command -v powerprofilesctl >/dev/null 2>&1; then
    powerprofilesctl list >> "$OUT" 2>&1 || echo "Could not get power profile info" >> "$OUT"
elif command -v upower >/dev/null 2>&1; then
    upower -i $(upower -e | grep 'BAT\|AC' | head -1) >> "$ARTIFACTS_DIR/power_info.txt" 2>&1 || true
    echo "Power info saved to artifacts directory" >> "$OUT"
else
    echo "Power management tools not available" >> "$OUT"
fi
echo "" >> "$OUT"

# NVMe Temperature (if available)
echo "=== NVMe Drive Temperature ===" >> "$OUT"
if command -v smartctl >/dev/null 2>&1; then
    for nvme in /dev/nvme*; do
        if [ -b "$nvme" ] && [ "$nvme" != "${nvme%[0-9]}" ]; then
            model=$(sudo smartctl -a "$nvme" 2>/dev/null | grep -i "model number" | cut -d: -f2 | xargs || echo "unknown")
            temp=$(sudo smartctl -a "$nvme" 2>/dev/null | grep -i "temperature" | head -1 | grep -oP '\d+' | head -1 || echo "N/A")
            echo "$nvme ($model): ${temp}°C" >> "$OUT" 2>&1 || true
        fi
    done
else
    echo "smartctl not available (install smartmontools)" >> "$OUT"
fi
echo "" >> "$OUT"

# Summary and Recommendations
echo "=== Summary ===" >> "$OUT"
echo "Diagnostic completed at $(date)" >> "$OUT"
echo "Review temperatures above:" >> "$OUT"
echo "  - CPU: ideal < 85°C, caution ≥ 90°C" >> "$OUT"
echo "  - GPU: ideal < 85°C" >> "$OUT"
echo "  - NVMe: ideal < 70°C" >> "$OUT"
echo "" >> "$OUT"
echo "If temperatures are high, consider:" >> "$OUT"
echo "  1. Limiting thread usage (OMP_NUM_THREADS, MKL_NUM_THREADS, etc.)" >> "$OUT"
echo "  2. Disabling Turbo Boost temporarily" >> "$OUT"
echo "  3. Lowering CPU frequency cap" >> "$OUT"
echo "  4. Switching to power-saver profile" >> "$OUT"
echo "" >> "$OUT"

echo "Report saved to: $OUT"
echo "Artifacts saved to: $ARTIFACTS_DIR"
echo ""
echo "To view the report:"
echo "  cat $OUT"
echo ""
echo "To clean up diagnostic files:"
echo "  rm $OUT"
echo "  rm -r $ARTIFACTS_DIR"

