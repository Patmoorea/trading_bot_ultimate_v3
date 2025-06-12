import torch
import time

def benchmark_m4():
    device = torch.device("mps")
    size = (5000, 5000)  # Taille réduite pour M4
    
    ops = {
        'matmul': lambda x: x @ x.T,
        'conv': lambda x: torch.conv2d(x.unsqueeze(0), x.unsqueeze(0)),
        'svd': lambda x: torch.svd(x)
    }
    
    x = torch.rand(size, device=device)
    
    for name, op in ops.items():
        try:
            start = time.time()
            op(x)
            torch.mps.synchronize()
            print(f"M4 {name}: {time.time()-start:.4f}s")
        except Exception as e:
            print(f"Erreur {name}: {str(e)}")

if __name__ == '__main__':
    benchmark_m4()
