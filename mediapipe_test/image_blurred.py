import cv2

img = cv2.imread('woo.jpg')

blurred_img = cv2.GaussianBlur(img, (5,5), 0)

cv2.imshow('Original', img)
cv2.imshow('Blurred', blurred_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

