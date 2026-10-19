import time
from pathlib import Path
from multiprocessing import Pool 
from functools import partial

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

def run_parallel(image_paths, worker, workers=None):
    """compare to run serial, runs in parallel"""
    start = time.perf_counter()
    results = []
    errors = []
    workerP = partial(_safe_call, worker) #this why we can call helper method within imap

    with Pool(processes = workers) as pool: #try/finally replacement better for files
        for output in pool.imap_unordered(workerP, image_paths): 
            if(output["success"]):
                results.append(output["data"])
            else:
                errors.append({"path": output["path"], "error": output["error"]})

    return {
            "results": results,
            "errors": errors,
            "total_seconds": time.perf_counter() - start,
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
