# pi-image-cluster
We are building a small compute cluster out of Raspberry Pis and using it to process a large batch of images in parallel.

## Requirements
- Python 3.12

## PIP Install Commands

Activate your virtual environment and install the required libraries:
```bash
pip install -r requirements.txt
```

## How to run it
The same small dataset of 100 as we used for testing (Only run once):
```bash
python scripts/build_dataset.py --num-images 100 --seed 42
```

Run from the project root:
```bash
python -m scripts.benchmark
```