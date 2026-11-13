import os
import time
from collections.abc import Callable, Sequence
from functools import partial
from multiprocessing import Pool
from multiprocessing.pool import Pool as PoolType
from pathlib import Path
from typing import Any

from cluster.worker import init_worker


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"} #maybe add more later maybe not


def find_images(folder: str | Path) -> list[str]:
    """
    Returns a sorted list of image file paths in a folder.

    Args:
        folder: Directory path to search for image files.

    Returns:
        A sorted list of string paths to matching file images.

    """

    image_paths = []

    for item in Path(folder).iterdir():
        extension = item.suffix.lower()

        if extension in IMAGE_EXTENSIONS:
            image_paths.append(str(item))

    image_paths.sort()
    return image_paths

def image_list(image_paths: Sequence[str | Path]) -> list[str]:
    """
    Grabs the base filenames from a sequence of full image paths.

    Args:
        image_paths: Sequence of file paths.

    Returns:
        A list of filename strings including the extensions in order.
    """

    best = []
    for path in image_paths:
        best.append(Path(path).name)

    return best

def run_serial(image_paths: Sequence[str | Path], worker: Callable[[str | Path], Any]) -> dict[str, Any]:
    """
    Process images one at a time as a single process baseline.

    Args:
        image_paths: Sequence of image file paths to process.
        worker: Callable function applied to each image path.

    Returns:
        A dictionary containing:
        - 'results' (dict[str, Any]): Mapping of image filename to worker output.
        - 'errors' (list[dict[str, str]]): List of errors for failed images.
        - 'total_seconds' (float): Total elapsed execution time.

    """
    start = time.perf_counter()
    results = {}
    errors = []

    for path in image_paths:
        try:

            file_name = Path(path).name
            results[file_name] = worker(path)

        except Exception as e:
            errors.append({"path": path, "error": str(e)})

    return {
        "results": results,
        "errors": errors,
        "total_seconds": time.perf_counter() - start,
    }






def _safe_call(worker: Callable[[str | Path], Any], path: str | Path) -> dict[str, Any]:
    """safely execute worker on path, catching any exceptions

    Args:
        worker: Callable function applied to the image path.
        path: File path to the target image.

    Returns:
        A dictionary which shows the execution status:
            - On success: '{"success": True, "data": <worker_output>, "path": path}'
            - On failure: '{"success": False, "data": <error_message>, "path": path}'

    """
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



def run_parallel(image_paths: Sequence[str | Path], worker: Callable[[str | Path], Any], workers: int | None = None, pool: PoolType | None = None) -> dict[str, Any]:
    """
    Processes images in parallel using multiprocess worker pool.

    Args:
        image_paths: Sequence of image file paths to process.
        worker: Callable function applied to each image path.
        workers: Number of worker processes to spawn if pool is None. Defaults to cpu count or 1.
        pool: Pre-initialized multiprocessing.pool instance to reuse on multiple benchmark runs.

    Returns:
        A dictionary containing:
            - 'results' (dict[str, Any]): Mapping of image filename to worker output.
            - 'errors' (list[dict[str, Any]]): List of error records for failed images.
            - 'total_seconds' (float): Elapsed execution time for the dispatch loop.


    """
    #
    results = {}
    errors = []
    workerP = partial(_safe_call, worker) #this why we can call helper method within imap

    actual_workers = workers or (os.cpu_count() or 1)

    chonky = max(1, len(image_paths) // (actual_workers * 4))


    def _execute(p: PoolType) -> float:
        start = time.perf_counter()
        for output in p.imap_unordered(workerP, image_paths, chunksize=chonky):
            if output["success"]:
                results[Path(output["path"]).name] = output["data"]

            else:
                errors.append({"path": output["path"], "error": output["error"]})
        return time.perf_counter() - start

    if pool is not None:
        elapsed = _execute(pool)

    else:

        with Pool(processes=workers, initializer=init_worker) as p:
            elapsed = _execute(p)

    return {
        "results": results,
        "errors": errors,
        "total_seconds": elapsed,
    }