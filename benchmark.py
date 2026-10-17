import csv
import os
import platform
import statistics
from datetime import datetime
from pathlib import Path

from cluster.driver import find_images, run_serial, run_parallel
from cluster.worker import process_image

REPEATS = 3


def time_serial(paths, worker, repeats):
    """run serial multiple times and return the median time"""
    times = []

    for i in range(repeats):
        out = run_serial(paths, worker)
        times.append(out["total_seconds"])

    return statistics.median(times)


def time_parallel(paths, worker, n_workers, repeats):
    """same, for run_parallel with a given worker count"""
    times = [
        run_parallel(paths, worker, workers=n_workers)["total_seconds"]
        for _ in range(repeats)
    ]
    return statistics.median(times)


def save_results(rows, out_dir):
    """save benchmark results to a CSV file"""
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
    """
    waiting on proper implementation of driver for accurate results
    note: when running BUNCH of popups
    
    """
    paths = find_images("images")

    worker_counts = {1, 2, 4, 8, os.cpu_count()}
    worker_counts = sorted(worker_counts)

    # 1. warm-up run
    print("Warm-up...")
    run_serial(paths, process_image)

    # 2. serial baseline
    print("Timing serial baseline...")
    baseline = time_serial(paths, process_image, REPEATS)

    # 3. benchmark each worker count
    rows = []

    print(f"\n{'Workers':<10}{'Time (s)':<12}{'Speedup':<10}{'Efficiency':<12}") #neat columns
    print("-" * 44)

    for w in worker_counts:
        t = time_parallel(paths, process_image, w, REPEATS)

        speedup = baseline / t #how many times faster parallel processing is than serial processing
        efficiency = speedup / w #speedup per worker, expressed as a percentage

        row = {
            "workers": w,
            "seconds": round(t, 4),
            "speedup": round(speedup, 2),
            "efficiency": round(efficiency, 2),
        }

        rows.append(row)

        print(f"{w:<10}{t:<12.3f}{speedup:<10.2f}{efficiency:<12.2%}") #headers

    # 4. save results to CSV
    save_results(rows, Path("results"))


if __name__ == "__main__":
    main()
