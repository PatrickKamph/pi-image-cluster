# Raspberry Pi Image Cluster

This project explores how image inference scales across multiple processes on
Raspberry Pi hardware. It uses Ultralytics YOLO to detect objects in images and
compares a serial baseline with multiprocessing runs using different worker
counts. 

## How it works

- `cluster.worker` loads `models/yolo11s.pt` and runs inference on an image.
- `cluster.driver` discovers `.jpg`, `.jpeg`, and `.png` images and provides
  serial and multiprocessing execution helpers. If no worker count is passed,
  it uses the host's available CPU count.
- `scripts.benchmark` warms up the model, measures serial and parallel runs,
  checks that each run processed the expected images, and writes timing results
  as a CSV file.

## Requirements

- Python 3.12 for the local environment 
- Dependencies listed in [`requirements.txt`](requirements.txt)

The `data/`, `models/`, and results folders are ignored by Git. Make sure the
images and model weights are present on the machine where you run the project.

## Setup

Create and activate a virtual environment, then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Prepare images

To download a reproducible sample of 100 unlabeled images that we used from the COCO 2017
dataset into `data/`, run:

```bash
python -m scripts.build_dataset --num-images 100 --seed 42
```

The script downloads the COCO image-info annotations when needed and skips
images that are already present. Use `--num-images 0` to download all available
images.

## Run a benchmark

From the project root, run:

```bash
python -m scripts.benchmark
```

The benchmark runs each worker-count configuration three times and reports
median elapsed time, speedup relative to the serial baseline, and efficiency.
CSV files are written to `results/`. `json_results/` is used by the detection
comparison output.
