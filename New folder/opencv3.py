import cv2
import cv2.cv2
import cv2.cv2

img = cv2.imread(r'C:\Users\Personal\Documents\projects\KICS Second Project\New folder\assets\logo.png', 1)
img= cv2.resize(img, (400, 400))
img = cv2.rotate(img, cv2.cv2.ROTATE)
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()