import torch
import platform

def check_gpu():
    try:
        if torch.backends.mps.is_available():
            print(f"[M4] GPU disponible | OS: {platform.mac_ver()[0]}")
        else:
            print("GPU non disponible")
    except Exception as e:
        print(f"Check GPU: {str(e)}")

if __name__ == "__main__":
    check_gpu()
