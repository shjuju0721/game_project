import cv2

import urllib.request

import os

xml_file = 'haarcascade_frontalface_default.xml'

if not os.path.exists(xml_file):
    url = f'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/{xml_file}'
    print("얼굴 인식 모델 파일을 다운로드 중입니다...")
    urllib.request.urlretrieve(url, xml_file)

face_cascade = cv2.CascadeClassifier(xml_file)
cap = cv2.VideoCapture(0)

while True:
    ret,frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors = 5, minSize=(30,30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (w,y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, 'face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow('Face Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


