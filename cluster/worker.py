import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO
import torch


# Should anchor the directory so that it wont output models folder outside of project directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "yolo11s.pt"

CONF_THRESHOLD = 0.2
IOU_THRESHOLD = 0.5


model: YOLO | None = None


def init_worker() -> None:
    """
    Initialize the YOLO model once per worker process and locks the thread counts to 1.

    Makes a dummy pass to run ultralytics' device setup, which then forces pyTorch threads back to 1 and disables
    OpenCV's multithreading to prevent CPU thread waste for all worker processes and ensures each worker iteration
    runs on the specified amount of cores and nothing more.


    """
    # This is just to make sure the root is properly initialized from YOLO
    global model

    cv2.setNumThreads(0)

    if model is None:
        model = YOLO(str(MODEL_PATH))
        # Run a dummy inference pass to force Ultralytics predictor setup and select_device()
        dummy = np.zeros((64, 64, 3), dtype=np.uint8)
        model(dummy, verbose=False)

    # Force PyTorch intra-op threads to 1 per worker process AFTER model/predictor initialization
    torch.set_num_threads(1)


def process_image(path: str | Path) -> list[dict[str, Any]]:
    """
    Runs YOLO object detection on a single image and returns its detections.

    Args:
        path: File path to the input image.

    Returns:
        A list of dictionaries representing detected objects, where each dictionary contains:
            - 'name' (str): Class label
            - 'confidence' (float): Detection confidence score rounded to 2 decimals.

    Raises:
        FileNotFoundError: If no image exists at the path

    """

    global model
    if model is None:
        init_worker()

    # Ensure thread count remains clamped to 1 for each image task
    if torch.get_num_threads() != 1:
        torch.set_num_threads(1)


    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found at path: {os.path.abspath(path)}")

    # The basic command, basically gets the image and lets u modify it
    image = cv2.imread(path)

    # Included verbose = False so console isn't cluttered. If you want to see all the deductions you can make it True.
    results = model(image, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False) #Change confidence value depending on the weights. Default was 0.25 but the model was having trouble with detecting the tree lol
    #IOU makes it so that u cant have two boxes too close to each other

    detections = []

    result = results[0]

    for box in result.boxes:
        subDict = {}

        cls = int(box.cls[0])
        conf = float(box.conf[0])
        name = model.names[cls]

        subDict['name'] = name
        subDict['confidence'] = round(conf, 2)

        detections.append(subDict)

    return detections