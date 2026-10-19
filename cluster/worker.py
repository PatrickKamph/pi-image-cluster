import os
import time
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO





def process_image(path: str) -> dict:

    model = YOLO("../models/yolo11s.pt")


    """temp function to work on driver"""

    start = time.perf_counter()

    # just incase there aint no image cuz for some reason it couldn't tell if there was an image or not
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found at path: {os.path.abspath(path)}")

    # The basic command, basically gets the image and lets u modify it
    image = cv2.imread(path)

    results = model(image, conf=0.2, iou=0.5) #Change confidence value depending on the weights. Default was 0.25 but the model was having trouble with detecting the tree lol
    #IOU makes it so that u cant have two boxes too close to each other

    result = results[0]


    # This is all to check if the model actually detected something cuz if it didnt it doesnt tell u :/
    if len(result.boxes) == 0:
        print("No objects in the image!")
    else:
        for box in result.boxes:
            cls = int(box.cls[0]) # cls means classification btw
            conf = float(box.conf[0]) # How sure it is that thing
            name = model.names[cls]
            print(f"Detected {name} w/ confidence {conf:.2f}") #Gonna round confidence to 2 digits #thank u docs :)



    """ To visualize results if needed for testing
    new_img = result.plot()
    cv2.imwrite('output.png', new_img)
    """



    #print(results)

    #plt.imshow(fin_img)
    #plt.axis("off")
    #plt.show()



    """ Blurring testing
    # bigger number = bigger blur
    k_size = 12

    blurred = cv2.blur(image, (k_size, k_size))

    print(blurred.size)
    """

    #This shows the image.
    #cv2.imshow('imagine', blurred)

    """
    This makes it so that it disappears once you press any key.
    You can make it go away after a few seconds (ex 5 seconds by replacing 0 with 5000
    Without this the image would not show fyi, not necessary when running on the
    raspberry pi but for testing its helpful.
    """
    #cv2.waitKey(0)



    time.sleep(0.1) #remove later its pretend work

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
