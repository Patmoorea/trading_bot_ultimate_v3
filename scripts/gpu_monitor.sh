#!/bin/zsh
# Monitor GPU/CPU usage
while true; do
    gpu_usage=$(sudo powermetrics --samplers gpu_power -n1 | grep "GPU Active")
    cpu_usage=$(top -l1 -s0 | grep "CPU usage")
    echo "$(date) | GPU: $gpu_usage | CPU: $cpu_usage" >> performance.log
    sleep 5
done
