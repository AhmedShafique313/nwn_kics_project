import cv2

img = cv2.imread(r'C:\Users\Personal\Documents\projects\KICS Second Project\New folder\assets\logo.png', 0)
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()