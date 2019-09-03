import cv2
import dlib
from imutils import face_utils, rotate
import numpy



eye_predictor = dlib.shape_predictor('eye_predictor.dat')

webcam = cv2.VideoCapture('test_vid.mp4')

while(True):
    # Capture frame-by-frame
    _, frame = webcam.read()
    frame = cv2.resize(rotate(frame, 270), None, fx=0.25, fy=0.25)[:,120:-120]
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame_rectangle = dlib.rectangle(left=0, top=0,
                                     right=frame.shape[1], bottom=frame.shape[0])
#
    shape = eye_predictor(gray_frame, frame_rectangle)
    shape = face_utils.shape_to_np(shape)

    for (x, y) in shape:
        cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
#
    mask = numpy.zeros(gray_frame.shape, numpy.uint8)
    cv2.fillPoly(mask, pts = [shape], color=(1,1,1))
    gray_frame = mask * gray_frame
#
    _, binary_frame = cv2.threshold(gray_frame, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

#
#    print(numpy.max(thresh1) )
#    print(numpy.min(thresh1))
#
#
    binary_frame = mask * binary_frame

    kernel = numpy.ones((5,5),numpy.uint8)
    binary_frame = cv2.erode(binary_frame,kernel,iterations = 1)
    _, contours, _ = cv2.findContours(binary_frame, 1, 2)
#
    for cnt in contours:
        if cnt.shape[0] > 5:
            (x,y),radius = cv2.minEnclosingCircle(cnt)
            center = (int(x),int(y))
            radius = int(radius)
            cv2.circle(frame,center,radius,(0,255,0),2)
#            ellipse = cv2.fitEllipse(cnt)
#            cv2.ellipse(frame, ellipse,(0,255,0),2)
##
##    circles = cv2.HoughCircles(frame,cv.HOUGH_GRADIENT,1,20,
##                    param1=50,param2=30,minRadius=0,maxRadius=0)
##    circles = numpy.uint16(numpy.around(circles))
##    for i in circles[0,:]:
##    # draw the outer circle
##        cv2.circle(frame,(i[0],i[1]),i[2],(0,255,0),2)
##    # draw the center of the circle
##        cv2.circle(frame,(i[0],i[1]),2,(0,0,255),3)
#

    cv2.imshow('frame', frame)
    cv2.waitKey(0)
    # Display the resulting frame
#    cv2.imshow('frame',frame)
#    cv2.waitKey(0)
#    if cv2.waitKey(10) & 0xFF == ord('q'):
#        break

# When everything done, release the capture
webcam.release()
cv2.destroyAllWindows()
