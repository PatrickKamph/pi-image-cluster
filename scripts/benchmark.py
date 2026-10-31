import csv
import os


"""
Sets threads to 1 before initializing YOLO stuff.
we NEED to have this at the top before we import torch or anything else, or else YOLO will ignore the
thread limit we set.
"""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)
#  Limit inter-op parallelism threads to 1 to eliminate background thread contention
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

import cv2
cv2.setNumThreads(0)

import platform
import statistics
from datetime import datetime
from pathlib import Path

from cluster.driver import find_images, run_serial, run_parallel, image_list
from cluster.worker import process_image, init_worker
from scripts.verification import compare_detections, record_final

from multiprocessing import Pool


REPEATS = 3
PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"


def extract_processed_paths(results):
    """helper method to extract path set from a run's results"""
    paths = set()

    for r in results:
        if isinstance(r, dict):
            path = r["path"]
        elif isinstance(r, tuple):
            path = r[0]
        else:
            path = r

        paths.add(str(path))

    return paths


def validate_run(out, paths, serial_paths=None):
    """checks errors, and compares if number of results matches number of paths"""

    results = out.get("results", [])
    err_list = out.get("errors", [])

    if err_list:
        return False, f'{len(err_list)} error(s) occurred'

    image_lists = image_list(paths)


    expected = len(image_lists)
    processed = set(results.keys())

    missing = expected - processed
    if missing:
        return False

    extra = processed - expected
    if extra:
        return False



    return True, "ok"


# Added serial_pats = None at the end so that it is forced to look at all 100 images. If it fails at doing that,
# it will give an error :)
def time_serial(paths, worker, repeats, serial_paths = None):
    """run serial multiple times and return the median time"""
    times = []

    # Ensure main process PyTorch intra-op threads are strictly locked to 1
    torch.set_num_threads(1)

    for i in range(repeats):
        out = run_serial(paths, worker)


        if serial_paths is None:
           serial_paths = extract_processed_paths(out.get("results", []))

        ok, reason = validate_run(out, paths, serial_paths)
        
        if not ok:
            raise RuntimeError(f"Fatal: invalid baseline on repeat {i + 1}: {reason}")

        times.append(out["total_seconds"])

    return statistics.median(times), serial_paths


def time_parallel(paths, worker, n_workers, repeats, serial_paths):
    """run parallal multiple times and return the median time"""
    times = []
    row_valid = True
    reasons = []


    # Made it stop calling run parallel over and over which destroyed the workers anytime it was done measuring for
    # that specific core count. Wont be too much of a change on the final portion on the pi's tho.
    with Pool(processes=n_workers, initializer=init_worker) as pool:
        for i in range(repeats):
            out = run_parallel(paths, worker, workers=n_workers, pool=pool)
            ok, reason = validate_run(out, paths, serial_paths)

            if not ok:
                row_valid = False
                reasons.append(f"repeat {i + 1}: {reason}")

            times.append(out["total_seconds"])

    return statistics.median(times), row_valid, "; ".join(reasons)


def save_results(rows, out_dir):
    """save benchmark results to a CSV file"""
    if not rows:
        print("\nNo valid rows to save")
        return
    
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") #4-digit year, 2-digit month, 2-digit day, an underscore, and the 2-digit hour, minute, and second
    machine = platform.node().replace(" ", "_") #grabs computer name but replaced space with underscore
    filename = out_dir / f"benchmark_{machine}_{timestamp}.csv" #output file name, name+timestamp from before

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()

        for row in rows:
            writer.writerow(row)

    print(f"\nSaved results to: {filename}")

def main():
    paths = find_images(IMAGE_DIR)

    if not paths:
        print("No images found")
        return

    max_workers = os.cpu_count() or 1
    worker_counts = {1, 2, 3, 4, 6, 8, 12, 16, max_workers}
    worker_counts = sorted(w for w in worker_counts if w <= max_workers)

    print("Warm-up...")
    # Initialize the model, execute the dummy warm-up, and clamp threads to 1
    # before running the serial baseline so baseline timing is a true single-threaded reference.
    init_worker()

    # Making a list of all image files taken from find_images. Since its sourced from find_images, all suffix's should
    # be only .jpg, .png, and .jpeg
    image_lists = image_list(paths)


    """
    compare_detections(serial_results, parallel_results, image_lists)

    Where would we insert comapre detections? How do we get the results?
    
    """

    run_serial(paths, process_image)


    print("Timing serial baseline...")
    baseline, serial_paths = time_serial(paths, process_image, REPEATS)

    print(f"Serial baseline: {baseline:.3f} seconds")


    rows = []

    print(
        f"\n{'Workers':<10}{'Time (s)':<12}"
        f"{'Speedup':<10}{'Efficiency':<12}{'Valid':<10}"
    )
    print("-" * 54)

    for w in worker_counts:
        t, row_valid, reason = time_parallel(paths, process_image, w, REPEATS, serial_paths)

        speedup = baseline / t if t > 0 else 0
        efficiency = speedup / w

        row = {
            "workers": w,
            "seconds": round(t, 4),
            "speedup": round(speedup, 2),
            "efficiency": round(efficiency, 2),
            "valid": row_valid,
            "validation_errors": reason,
        }

        rows.append(row)

        print(
            f"{w:<10}{t:<12.3f}{speedup:<10.2f}"
            f"{efficiency:<12.2%}{str(row_valid):<10}"
        )

        if not row_valid:
            print(f"  Problem: {reason}")


    save_results(rows, RESULTS_DIR)


if __name__ == "__main__":
    main()
