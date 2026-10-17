import os
import time
import cv2


def process_image(path: str) -> dict:
    """temp function to work on driver"""

    # The basic command, basically gets the image and lets u modify it
    image = cv2.imread(path)

    # bigger number = bigger blur
    k_size = 12
    blurred = cv2.blur(image, (k_size, k_size))




    #This shows the image.
    cv2.imshow('imagine', blurred)

    """
    This makes it so that it disappears once you press any key.
    You can make it go away after a few seconds (ex 5 seconds by replacing 0 with 5000
    Without this the image would not show fyi, not necessary when running on the
    raspberry pi but for testing its helpful.
    """
    cv2.waitKey(0)



    start = time.perf_counter()
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
