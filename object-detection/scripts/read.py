import cv2 as cv

# Rescales the given frame for the given scale 
# Works on live video, prerecorded videos, feeds and images
def rescaleFrame(frame, scale = 0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

# Changes resolution to the given width and height
# Works only on live videos
def changeResolution(width, height):
    capture.set(3, width)
    capture.set(4, height)


#Reading video
capture = cv.VideoCapture("../media/video/1.mp4")
while True:
    isTrue, frame = capture.read();

    frame_resized = rescaleFrame(frame)

    cv.imshow('Video', frame)
    cv.imshow('Video resized', frame_resized)

    if cv.waitKey(20) & 0xFF == ord('d'):
        break
    
capture.release()
cv.destroyAllWindows()
