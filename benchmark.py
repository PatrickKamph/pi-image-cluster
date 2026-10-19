import csv
import os
import platform
import statistics
from datetime import datetime
from pathlib import Path

from cluster.driver import find_images, run_serial, run_parallel
from cluster.worker import process_image

REPEATS = 3
ON_VALID = "flag" 

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
    errors = out.get("errors", [])

    if len(errors) > 0:
        message = str(len(errors)) + " error(s) occurred"
        return False, message

    total = len(results) + len(errors)

    if total != len(paths):
        message = ("count mismatch (" + str(len(results)) + " + " + str(len(errors)) + " != " + str(len(paths)) + ")")
        return False, message

    processed = extract_processed_paths(results)


    if serial_paths is not None:
        expected = serial_paths
    else:
        expected = set()

        for p in paths:
            expected.add(str(p))

    if processed != expected:
        return False, "processed paths do not match serial run"

    return True, "ok"

def time_serial(paths, worker, repeats, serial_paths = None):
    """run serial multiple times and return the median time"""
    times = []

    for i in range(repeats):
        out = run_serial(paths, worker)
        ok, reason = validate_run(out, paths, serial_paths)
        
        if not ok:
            raise RuntimeError(f"Fatal: invalid baseline on repeat {i + 1}: {reason}")
        if serial_paths is None:
            serial_paths = extract_processed_paths(out.get("results", []))

        times.append(out["total_seconds"])

    return statistics.median(times), serial_paths


def time_parallel(paths, worker, n_workers, repeats, serial_paths):
    """run parallal multiple times and return the median time"""
    times = []
    row_valid = True
    reasons = []

    for i in range(repeats):
        out = run_parallel(paths, worker, workers=n_workers)
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
    paths = find_images("images")

    if not paths:
        print("No images found")
        return

    worker_counts = {1, 2, 4, 8, os.cpu_count() or 1}
    worker_counts = sorted(worker_counts)

    print("Warm-up...")
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


    save_results(rows, Path("results"))


if __name__ == "__main__":
    main()
