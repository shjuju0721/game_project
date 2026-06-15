import cv2
import numpy as np
import tensorflow as tf

MODEL_FILE = "puff_model.keras"
IMG_SIZE = (160, 160)   # 학습 때와 똑같은 크기여야 함

# 학습 때 class_names 순서가 ['neutral', 'puff'] 였음 (알파벳 순)
# 모델 출력은 0~1 사이 한 개 값 = 'puff일 확률'
CLASS_NAMES = ["neutral", "puff"]

model = tf.keras.models.load_model(MODEL_FILE)
print("모델 불러오기 완료!")

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    # ─── 모델에 넣을 형태로 변환 ───
    # 1) 학습 때 크기(160x160)로 리사이즈
    img = cv2.resize(frame, IMG_SIZE)
    # 2) OpenCV는 BGR, 학습은 RGB였으므로 색 순서 변환
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 3) 모델은 여러 장(batch) 단위를 기대하므로 차원 하나 추가: (160,160,3) → (1,160,160,3)
    #    ※ 픽셀값 0~255 그대로 넣음. 학습 코드 안에 Rescaling 층이 있어서 자동 변환됨.
    input_data = np.expand_dims(img.astype(np.float32), axis=0)

    # ─── 예측 ───
    prob_puff = model.predict(input_data, verbose=0)[0][0]  # puff일 확률 (0~1)

    # 0.5 기준으로 판정
    if prob_puff >= 0.5:
        label = "PUFF"
        confidence = prob_puff
        color = (0, 165, 255)
    else:
        label = "NEUTRAL"
        confidence = 1 - prob_puff
        color = (0, 255, 0)

    # ─── 화면 표시 ───
    text = f"{label}  ({confidence:.0%})"
    cv2.putText(frame, text, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)
    # 확률 막대 (puff 쪽으로 얼마나 기울었나 시각화)
    bar_len = int(prob_puff * 300)
    cv2.rectangle(frame, (10, 70), (10 + bar_len, 90), (0, 165, 255), -1)
    cv2.rectangle(frame, (10, 70), (310, 90), (255, 255, 255), 2)

    cv2.imshow("Puff Test", cv2.flip(frame, 1))

    if cv2.waitKey(5) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()