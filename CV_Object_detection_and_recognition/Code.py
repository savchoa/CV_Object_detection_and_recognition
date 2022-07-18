import numpy as np
import cv2
import time

fourcc = cv2.VideoWriter_fourcc(*'XVID')
frameSize = (1920, 1080)
out = cv2.VideoWriter('output_video.avi', fourcc, 30, frameSize)

cap = cv2.VideoCapture('task_video.mp4')
frame_counter = 1


def detect_shape(c):
    shape = "unidentified"
    if len(c) == 3:
        shape = "triangle"
    elif len(c) == 4:
        (center, (width, height), angle) = cv2.minAreaRect(c)
        ar = width / height
        shape = "square" if ar >= 0.65 and ar <= 1.35 else "rectangle"
    elif len(c) >= 7:
        shape = "circle"
    for point in c:
        if (point[0,0] == 0) or (point[0,0] == frame.shape[1]-1) \
                or (point[0,1] == 0) or (point[0,1] == frame.shape[0]-1):
            shape = "unidentified"

    return shape

def show_shape(shape):
    cv2.rectangle(frame, (x-10, y-10), (x + w+20, y + h+20), shape_color[shape], 3)
    # cv2.drawContours(frame, [approx], -1, (0, 255, 0), 3)
    cv2.putText(frame, shape, (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, shape_color[shape])


shape_color = {"unidentified": (0, 0, 0), "triangle": (255, 0, 0),
               "square": (0, 255, 255), "rectangle": (0, 125, 0),
               "circle": (0, 0, 255)}

while True:
    _, frame = cap.read()
    if frame is None:
        break

    frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_lim = np.array([30,40,140])
    upper_lim = np.array([100,110,240])
    mask = cv2.inRange(frame_HSV, lower_lim, upper_lim)
    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(mask, kernel, iterations=6)
    dilation = cv2.dilate(erosion, kernel, iterations=2)
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        area = cv2.contourArea(approx)
        x, y, w, h = cv2.boundingRect(approx)
        if ((area > 5500) & (frame_counter < 550) & (len(approx) <= 10)&(x<1400)) or \
                ((frame_counter >= 550) & (area > 2000) & (y<900)):
            shape = detect_shape(approx)
            show_shape(shape)


    out.write(frame)
    cv2.imshow('Object_detection_and_recognition', frame)
    if cv2.waitKey(3) == ord('q'):
        break
    # time.sleep(0.1)
    frame_counter +=1

out.release()
cap.release()
cv2.destroyAllWindows()