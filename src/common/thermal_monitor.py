#!/usr/bin/env python3
"""
Thermal Monitoring Utility for ML Workloads

Monitors CPU, GPU, and system temperature during or after ML code execution.
Provides real-time updates and generates alerts for high temperatures.
"""

import subprocess
import time
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List, Tuple


class ThermalMonitor:
    """Monitor system thermal state and provide recommendations."""
    
    def __init__(self, alert_cpu: float = 85.0, alert_gpu: float = 80.0):
        """
        Initialize thermal monitor.
        
        Args:
            alert_cpu: CPU temperature threshold for alerts (°C)
            alert_gpu: GPU temperature threshold for alerts (°C)
        """
        self.alert_cpu = alert_cpu
        self.alert_gpu = alert_gpu
        self.history: List[Dict] = []
        
    def get_cpu_temp(self) -> Optional[float]:
        """Get CPU package temperature from sensors."""
        try:
            result = subprocess.run(
                ['sensors'],
                capture_output=True,
                text=True,
                timeout=5
            )
            # Parse output for Package id 0 or similar
            lines = result.stdout.split('\n')
            for line in lines:
                if 'Package id 0' in line or 'Tdie' in line or 'Tctl' in line:
                    # Extract temperature value
                    import re
                    match = re.search(r'\+?(-?\d+\.?\d*)°C', line)
                    if match:
                        return float(match.group(1))
                # Fallback: look for any temperature reading
                if '°C' in line and ('high' in line.lower() or 'crit' in line.lower()):
                    match = re.search(r'\+?(-?\d+\.?\d*)°C', line)
                    if match:
                        temp = float(match.group(1))
                        if 20 < temp < 150:  # Sanity check
                            return temp
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
            pass
        
        # Fallback: try thermal zones
        try:
            for zone_path in Path('/sys/class/thermal').glob('thermal_zone*/temp'):
                temp = int(zone_path.read_text().strip()) / 1000.0
                type_file = zone_path.parent / 'type'
                if type_file.exists():
                    zone_type = type_file.read_text().strip()
                    if 'cpu' in zone_type.lower() or 'package' in zone_type.lower():
                        if 20 < temp < 150:
                            return temp
        except (OSError, ValueError):
            pass
        
        return None
    
    def get_gpu_temp(self) -> Optional[float]:
        """Get GPU temperature from nvidia-smi."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                return float(result.stdout.strip())
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
            pass
        return None
    
    def get_cpu_freq(self) -> Optional[float]:
        """Get current CPU frequency in GHz."""
        try:
            result = subprocess.run(
                ['grep', 'cpu MHz', '/proc/cpuinfo'],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                freqs = [float(line.split(':')[1].strip()) for line in result.stdout.split('\n') if 'cpu MHz' in line]
                if freqs:
                    return max(freqs) / 1000.0  # Convert to GHz
        except (subprocess.TimeoutExpired, ValueError):
            pass
        return None
    
    def get_turbo_status(self) -> Optional[bool]:
        """Check if Turbo Boost is enabled."""
        try:
            turbo_file = Path('/sys/devices/system/cpu/intel_pstate/no_turbo')
            if turbo_file.exists():
                return turbo_file.read_text().strip() == '0'
        except OSError:
            pass
        return None
    
    def check_throttling(self) -> Optional[str]:
        """Check for thermal throttling in kernel logs."""
        try:
            result = subprocess.run(
                ['journalctl', '-k', '--since', '-1 hours', '--no-pager'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                throttling_messages = [
                    line for line in result.stdout.split('\n')
                    if any(term in line.lower() for term in ['throttl', 'thermal', 'overheat'])
                ]
                if throttling_messages:
                    return throttling_messages[-1]  # Return most recent
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return None
    
    def get_status(self) -> Dict:
        """Get current thermal status."""
        status = {
            'timestamp': datetime.now().isoformat(),
            'cpu_temp': self.get_cpu_temp(),
            'gpu_temp': self.get_gpu_temp(),
            'cpu_freq': self.get_cpu_freq(),
            'turbo_enabled': self.get_turbo_status(),
        }
        
        # Check for alerts
        alerts = []
        if status['cpu_temp'] is not None and status['cpu_temp'] >= self.alert_cpu:
            alerts.append(f"High CPU temperature: {status['cpu_temp']:.1f}°C (threshold: {self.alert_cpu}°C)")
        if status['gpu_temp'] is not None and status['gpu_temp'] >= self.alert_gpu:
            alerts.append(f"High GPU temperature: {status['gpu_temp']:.1f}°C (threshold: {self.alert_gpu}°C)")
        
        status['alerts'] = alerts
        status['throttling'] = self.check_throttling()
        
        return status
    
    def monitor(self, interval: float = 2.0, duration: Optional[float] = None, output_file: Optional[Path] = None):
        """
        Monitor system temperature continuously.
        
        Args:
            interval: Time between measurements (seconds)
            duration: Total monitoring duration (None for infinite)
            output_file: Optional file to save history
        """
        start_time = time.time()
        print(f"Starting thermal monitoring (interval: {interval}s)")
        print(f"Alerts: CPU ≥ {self.alert_cpu}°C, GPU ≥ {self.alert_gpu}°C")
        print("-" * 60)
        
        try:
            while True:
                status = self.get_status()
                self.history.append(status)
                
                # Print status
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"[{timestamp}] ", end="")
                
                if status['cpu_temp'] is not None:
                    cpu_icon = "⚠️" if status['cpu_temp'] >= self.alert_cpu else "✓"
                    print(f"CPU: {status['cpu_temp']:.1f}°C {cpu_icon}", end="")
                else:
                    print("CPU: N/A", end="")
                
                if status['gpu_temp'] is not None:
                    gpu_icon = "⚠️" if status['gpu_temp'] >= self.alert_gpu else "✓"
                    print(f" | GPU: {status['gpu_temp']:.1f}°C {gpu_icon}", end="")
                else:
                    print(" | GPU: N/A", end="")
                
                if status['cpu_freq'] is not None:
                    print(f" | Freq: {status['cpu_freq']:.2f} GHz", end="")
                
                print()
                
                # Print alerts
                if status['alerts']:
                    for alert in status['alerts']:
                        print(f"  ⚠️  {alert}")
                
                if status['throttling']:
                    print(f"  ⚠️  Throttling detected: {status['throttling'][:80]}")
                
                # Check duration
                if duration and (time.time() - start_time) >= duration:
                    break
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
        
        # Save history if requested
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(self.history, f, indent=2)
            print(f"\nHistory saved to: {output_file}")
        
        # Print summary
        if self.history:
            self._print_summary()
    
    def _print_summary(self):
        """Print summary statistics."""
        cpu_temps = [h['cpu_temp'] for h in self.history if h['cpu_temp'] is not None]
        gpu_temps = [h['gpu_temp'] for h in self.history if h['gpu_temp'] is not None]
        
        print("\n" + "=" * 60)
        print("Monitoring Summary")
        print("=" * 60)
        
        if cpu_temps:
            print(f"CPU Temperature: min={min(cpu_temps):.1f}°C, max={max(cpu_temps):.1f}°C, avg={sum(cpu_temps)/len(cpu_temps):.1f}°C")
        if gpu_temps:
            print(f"GPU Temperature: min={min(gpu_temps):.1f}°C, max={max(gpu_temps):.1f}°C, avg={sum(gpu_temps)/len(gpu_temps):.1f}°C")
        
        alert_count = sum(len(h.get('alerts', [])) for h in self.history)
        if alert_count > 0:
            print(f"\n⚠️  Total alerts: {alert_count}")
            print("Consider applying thermal mitigations (see docs/training/THERMAL_MANAGEMENT.md)")


def main():
    """Command-line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor system thermal state during ML workloads')
    parser.add_argument('--interval', type=float, default=2.0, help='Update interval in seconds')
    parser.add_argument('--duration', type=float, help='Monitoring duration in seconds (default: infinite)')
    parser.add_argument('--alert-cpu', type=float, default=85.0, help='CPU temperature alert threshold (°C)')
    parser.add_argument('--alert-gpu', type=float, default=80.0, help='GPU temperature alert threshold (°C)')
    parser.add_argument('--output', type=Path, help='Save monitoring history to JSON file')
    parser.add_argument('--once', action='store_true', help='Take a single measurement and exit')
    
    args = parser.parse_args()
    
    monitor = ThermalMonitor(alert_cpu=args.alert_cpu, alert_gpu=args.alert_gpu)
    
    if args.once:
        status = monitor.get_status()
        print(json.dumps(status, indent=2))
        if status['alerts']:
            sys.exit(1)
    else:
        monitor.monitor(
            interval=args.interval,
            duration=args.duration,
            output_file=args.output
        )


if __name__ == '__main__':
    main()

