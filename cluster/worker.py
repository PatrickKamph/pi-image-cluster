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



# Fix to ensure threads are actually set to 0 since it kept being reset back.
# Without this initalization the threads arent properly set to one for each worker drone, which ruins the time.
def init_worker():
    # This is just to make sure the root is properly initialized from the yolo stuff
    global model


    torch.set_num_threads(1)
    cv2.setNumThreads(0)

    if model is None:
        model = YOLO(str(MODEL_PATH))


def process_image(path: str) -> dict:
    """temp function to work on driver"""

    global model
    if model is None:
        init_worker()



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
        "result": os.path.getsize(path),
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
