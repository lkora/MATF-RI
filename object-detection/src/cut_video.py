import cv2 as cv
import numpy as np
import os

file = "3.mp4"

# Playing video from file:
capture = cv.VideoCapture('video/' + file)

try:
    if not os.path.exists('data'):
        os.makedirs('data')
        os.makedirs('data/train')
        os.makedirs('data/test')
except OSError:
    print ('Error: Creating a directory')

currentFrame = 0
while(True):
    # Capture frame-by-frame
    isTrue, frame = capture.read()

    # Saves image of the current frame in jpg file
    if currentFrame % 1000 == 0:
        # Every 10th frame is 
        name = './data/test/frame' + '_' + file +  '_' + str(currentFrame) + '.jpg'
        print ('Creating...' + name)
        cv.imwrite(name, frame)
        # To stop duplicate images
    elif currentFrame % 100 == 0:
        name = './data/train/frame'  + '_' + file +  '_' + str(currentFrame) + '.jpg'
        print ('Creating...' + name)
        cv.imwrite(name, frame)

    currentFrame += 1


# When everything done, release the capture
capture.release()
cv.destroyAllWindows()