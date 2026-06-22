#!/bin/bash
# Memory monitoring script for sn45_vali
LOG_FILE="/home/rizzo/alpharidge-ai/validator_memory.log"
echo "=== Validator Memory Monitoring Started $(date) ===" >> "$LOG_FILE"
echo "Timestamp,Uptime,Memory_MB,CPU" >> "$LOG_FILE"

while true; do
    # Get pm2 info for sn45_vali
    PM2_INFO=$(pm2 jlist 2>/dev/null | python3 -c "
import sys, json
data = json.load(sys.stdin)
for p in data:
    if p.get('name') == 'sn45_vali':
        mem_mb = p.get('monit', {}).get('memory', 0) / 1024 / 1024
        cpu = p.get('monit', {}).get('cpu', 0)
        uptime = p.get('pm2_env', {}).get('pm_uptime', 0)
        import time
        uptime_min = int((time.time() * 1000 - uptime) / 1000 / 60) if uptime else 0
        print(f'{uptime_min},{mem_mb:.1f},{cpu}')
        break
")
    if [ -n "$PM2_INFO" ]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S'),$PM2_INFO" >> "$LOG_FILE"
        echo "$(date '+%H:%M:%S') - Uptime: $(echo $PM2_INFO | cut -d',' -f1)min, Mem: $(echo $PM2_INFO | cut -d',' -f2)MB"
    fi
    sleep 60
done
