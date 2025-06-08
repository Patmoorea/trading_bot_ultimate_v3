import time
from datetime import datetime


def trace_execution(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        print(f"[{datetime.now()}] EXEC {func.__name__}")
        result = func(*args, **kwargs)
        print(f"[{datetime.now()}] DONE {func.__name__} in {time.time()-start:.2f}s")
        return result

    return wrapper
