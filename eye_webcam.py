import cv2
_, binary_image = cv2.threshold(grayscaled, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
