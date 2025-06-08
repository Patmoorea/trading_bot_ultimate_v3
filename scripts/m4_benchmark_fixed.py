import torch
import time
from src.core_merged.mps_fixed import device

def safe_benchmark():
    try:
        size = (3000, 3000)  # Taille réduite pour stabilité
        x = torch.rand(size, device=device)
        
        start = time.time()
        _ = x @ x.T
        torch.mps.synchronize()
        print(f"MatMul réussi en {time.time()-start:.4f}s")
        
        return True
    except Exception as e:
        print(f"Échec benchmark: {str(e)}")
        return False

if __name__ == '__main__':
    safe_benchmark()
