import time
import tracemalloc
from core import main  # Adaptez à votre structure réelle

def run_audit():
    tracemalloc.start()
    
    start_time = time.time()
    snapshot1 = tracemalloc.take_snapshot()
    
    # Exécutez une itération complète
    main.run_cycle()  # À adapter à votre fonction principale
    
    snapshot2 = tracemalloc.take_snapshot()
    end_time = time.time()
    
    # Analyse mémoire
    top_stats = snapshot2.compare_to(snapshot1, 'lineno')
    print("[ Mémoire ] Top allocations:")
    for stat in top_stats[:5]:
        print(stat)
    
    # Analyse temps
    print(f"\n[ Performance ] Temps total: {end_time-start_time:.2f}s")

if __name__ == "__main__":
    run_audit()
