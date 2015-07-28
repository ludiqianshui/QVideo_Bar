import cv2
import cv2.cv as cv
# Load an color image in grayscale
img = cv2.imread('1.jpg',0)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()