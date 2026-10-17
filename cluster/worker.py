import os
import time


def process_image(path):
    """temp function to work on driver"""
    
    start = time.perf_counter()
    time.sleep(0.1) #remove later its pretend work

    return {
        "path": path,
        "result": os.path.getsize(path),
        "seconds": time.perf_counter() - start,
    }