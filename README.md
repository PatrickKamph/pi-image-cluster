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

put images in images/, then run from the project root
```bash
python -m cluster.driver
```