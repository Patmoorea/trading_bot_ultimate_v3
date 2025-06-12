#!/bin/bash

# ===== TRADING BOT M4 MONITOR =====
echo "\n=== System Hardware ==="
system_profiler SPHardwareDataType | grep -E "Chip|Memory"

echo "\n=== Memory Usage ==="
vm_stat | awk '{if(NR==1) print; if(NR==2||NR==3) print}'

echo "\n=== TensorFlow Devices ==="
python -c "import tensorflow as tf; print('\n'.join([f'- {d.device_type}: {d.name}' for d in tf.config.get_visible_devices()]))"

echo "\n=== GPU Performance ==="
metalperf 2>/dev/null || echo "Metal Performance Stats not available"
