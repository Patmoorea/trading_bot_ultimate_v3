import subprocess
import re

def get_gpu_temp():
    """Récupère la température du GPU en °C"""
    try:
        result = subprocess.run(['sudo', 'powermetrics', '--samplers', 'smc', '-n', '1'],
                              capture_output=True, text=True)
        match = re.search(r'GPU die temperature: (\d+\.\d+) C', result.stdout)
        return float(match.group(1)) if match else None
    except:
        return None
