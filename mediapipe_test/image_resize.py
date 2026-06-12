import cv2

img = cv2.imread('woo.jpg')

#1. 이미지 크기 조절(가로 400, 세로 300 픽셀로 고정)
resized_img = cv2.resize(img, (400, 300))

#2. 이미지 크기 조절(비율로 조절 - 0.5배)
resized_ratio = cv2.resize(img, (0,0), fx=0.5, fy=0.5)

#3. 이미지 자르기(clipped / crop)
#NumPy 배열 슬라이싱 사용[Y축 시작: y축 끝, x축 시작: x축 끝]
#예 : y축 100~400, x 축 200~500 영역 자르기
cropped_img = img[100:400, 200:500]

cv2.imshow('Resized', resized_img) #사진을 보여달라는 함수
cv2.imshow('Cropped', cropped_img)
cv2.waitKey(0)  #0번 누를때까지 기다려 달라 0번 누르면 끝
cv2.destroyAllWindows()  #해제 시켜 주는 것
