#!/usr/bin/env python3
from core import GPU_AVAILABLE, check_gpu_status

def main():
    print(f"=== Statut GPU ===")
    print(f"Configuration initiale: {GPU_AVAILABLE}")
    print(f"Vérification en temps réel: {check_gpu_status()}")

if __name__ == "__main__":
    main()
