import os
import time
from pathlib import Path
from multiprocessing import Pool 
from functools import partial
from cluster.worker import init_worker


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"} #maybe add more later maybe not


def find_images(folder):
    """return a sorted list of image paths in a folder"""
    image_paths = []

    for item in Path(folder).iterdir():
        extension = item.suffix.lower()

        if extension in IMAGE_EXTENSIONS:
            image_paths.append(str(item))

    image_paths.sort()
    return image_paths

def run_serial(image_paths, worker):
    """process images one at a time, this is the baseline"""
    start = time.perf_counter()
    results = []
    errors = []

    for path in image_paths:
        try:
            results.append(worker(path))
        except Exception as e:
            errors.append({"path": path, "error": str(e)})

    return {
        "results": results,
        "errors": errors,
        "total_seconds": time.perf_counter() - start,
    }

def _safe_call(worker, path):
    """safely execute worker on path, catching any exceptions"""
    try:
        return {
            "success": True, 
            "data": worker(path), 
            "path": path
        }
    except Exception as e:
        return {
            "success": False, 
            "error": str(e), 
            "path": path
        }



def run_parallel(image_paths, worker, workers=None, pool = None):
    """compare to run serial, runs in parallel"""
    #
    results = []
    errors = []
    workerP = partial(_safe_call, worker) #this why we can call helper method within imap

    actual_workers = workers or 1

    # need ts for the pi's cuz they got teeny tiny cores
    chonky = max(1, len(image_paths) // (workers * 4))

    """
        
    Makes it so that instead of taking one image at a time, it gonna send a bundle of images. Helps cuz it means there's
    less processing (runs every time it wants a new image before). Trying every method to save time :fire: :aura: :sandrone:
    
    
    
    10/23/2026 - Yanis
    I changed this the most here tbh
    
    The _execute function/method allows benchmark to pass the photos in the same pool instead of making a new one
    every time, which I think cut down speeds? probably. It cut down the processes at least. Half this stuff was on
    random stackoverflow boards.
    
    Also, I changed the start to be inside the execute function instead of outside, since it was including the amount of time
    it took to initialize the model and the latency.
    """

    def _execute(p):
        start = time.perf_counter()
        for output in p.imap_unordered(workerP, image_paths, chunksize=chonky):
            if output["success"]:
                results.append(output["data"])
            else:
                errors.append({"path": output["path"], "error": output["error"]})
        return time.perf_counter() - start

    if pool is not None:
        elapsed = _execute(pool)
    else:

        """
        This is honestly the most important part. Since its being initialized with init_worker (from worker.py),
        all the threads are actually being locked to one per worker, so when being run, it isnt going back to the YOLO
        base settings of taking ur CPU's best cores. This actually makes it so that we are parallel processing
        everything! It also makes it so we arent reinitializing the model and the threads every time we call the worker
        file.
        """
        with Pool(processes=workers, initializer=init_worker) as p:
            elapsed = _execute(p)

    return {
        "results": results,
        "errors": errors,
        "total_seconds": elapsed,
    }




if __name__ == "__main__":
    from cluster.worker import process_image

    paths = find_images("images")

    #serial
    out = run_serial(paths, process_image)
    print(
        f"{len(out['results'])} done, {len(out['errors'])} errors, "
        f"{out['total_seconds']:.2f}s"
    )

    #parallel

    out_para = run_parallel(paths, process_image)
    print(
        f"{len(out_para['results'])} done, {len(out_para['errors'])} errors, "
        f"{out_para['total_seconds']:.2f}s"
        )
