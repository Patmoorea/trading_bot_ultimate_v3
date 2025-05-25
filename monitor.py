import time
import psutil
import tensorflow as tf

def monitor_system():
    print("=== MONITORING EN TEMPS RÉEL ===")
    while True:
        # Performances GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            mem = tf.config.experimental.get_memory_info('GPU:0')
            print(f"GPU: {mem['current']/1e9:.2f}GB / {mem['peak']/1e9:.2f}GB")
        
        # Performances système
        print(f"CPU: {psutil.cpu_percent()}% | RAM: {psutil.virtual_memory().percent}%")
        time.sleep(5)

if __name__ == "__main__":
    monitor_system()
