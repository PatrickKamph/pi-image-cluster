import os
import time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import torch


# Should anchor the directory now so that it wont output models folder outside of project directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "yolo11s.pt"

# Thanks python type hints
model: YOLO | None = None


# Fix to ensure threads are actually set to 1 per worker.
# Ultralytics lazily calls select_device() during the first inference call,
# which internally overrides torch.set_num_threads to min(8, cpu_count - 1) (8 threads on this CPU).
# We execute a single warm-up pass with dummy data so lazy initialization happens immediately,
# and then clamp torch.set_num_threads back to 1. This prevents massive CPU thread oversubscription probably
def init_worker():
    # This is just to make sure the root is properly initialized from the yolo stuff
    global model

    cv2.setNumThreads(0)

    if model is None:
        model = YOLO(str(MODEL_PATH))
        # Run a dummy inference pass to force Ultralytics predictor setup and select_device()
        dummy = np.zeros((64, 64, 3), dtype=np.uint8)
        model(dummy, verbose=False)

    # Force PyTorch intra-op threads to 1 per worker process AFTER model/predictor initialization
    torch.set_num_threads(1)


def process_image(path: str) -> dict:
    """temp function to work on driver"""

    global model
    if model is None:
        init_worker()

    # Ensure thread count remains clamped to 1 for each image task
    if torch.get_num_threads() != 1:
        torch.set_num_threads(1)

    start = time.perf_counter()

    # just incase there aint no image cuz for some reason it couldn't tell if there was an image or not
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found at path: {os.path.abspath(path)}")

    # The basic command, basically gets the image and lets u modify it
    image = cv2.imread(path)

    # Included verbose = False so console isn't cluttered. If you want to see all the deductions you can make it True.
    results = model(image, conf=0.2, iou=0.5, verbose=False) #Change confidence value depending on the weights. Default was 0.25 but the model was having trouble with detecting the tree lol
    #IOU makes it so that u cant have two boxes too close to each other



    # This is all to check if the model actually detected something cuz if it didnt it doesnt tell u :/
    # Commented out since it just increases run time for no reason
    """
    if len(result.boxes) == 0:
        print("No objects in the image!")
    else:
        for box in result.boxes:
            cls = int(box.cls[0]) # cls means classification btw
            conf = float(box.conf[0]) # How sure it is that thing
            name = model.names[cls]
            print(f"Detected {name} w/ confidence {conf:.2f}") #Gonna round confidence to 2 digits #thank u docs :)
    """


    return {
        "path": path,
        "result": os.path.getsize(path), #REMOVE SOON
        "seconds": time.perf_counter() - start,
    }


""" This was for testing. Didnt want to touch the driver file to test this.
Theoretically, you would run a for loop in driver with each directory,
and run worker.process_image(filepath). Just gotta find a way to
allocate images to each worker yk.


if __name__ == '__main__':

    img_path = os.path.join('..', 'images', 'image1.jpg')

    process_image(img_path)
"""
