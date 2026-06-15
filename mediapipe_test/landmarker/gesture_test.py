# ============================================================
#  동작 테스트 (표정 + 턱 당기기)
# ============================================================

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "C:/Users/sinhj/embeded/game/face_landmarker.task"

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,   # ★ 머리 각도용 옵션 켜기
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)
cap = cv2.VideoCapture(0)

print("=" * 50)
print("각 동작을 해보세요. 턱 당기기(7번)도 확인.")
print("종료: q")
print("=" * 50)

# 기준 각도 저장용 (정면 볼 때의 값)
base_pitch = None

while True:
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    if result.face_blendshapes:
        scores = {b.category_name: b.score for b in result.face_blendshapes[0]}

        gestures = {
            "1.Open":   scores.get("jawOpen", 0),
            "2.Smile":  (scores.get("mouthSmileLeft", 0) + scores.get("mouthSmileRight", 0)) / 2,
            "3.Pucker": scores.get("mouthPucker", 0),
            "6.Press":  (scores.get("mouthPressLeft", 0) + scores.get("mouthPressRight", 0)) / 2,
        }

        y = 50
        for name, score in gestures.items():
            color = (0, 255, 0) if score > 0.4 else (200, 200, 200)
            cv2.putText(frame, f"{name}: {score:.2f}", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            y += 45

        # ─── 턱 당기기 (머리 위아래 각도 pitch) ───
        if result.facial_transformation_matrixes:
            matrix = np.array(result.facial_transformation_matrixes[0])
            # 회전 행렬에서 pitch(위아래 끄덕임) 각도 추출
            pitch = np.degrees(np.arctan2(-matrix[2][1], matrix[2][2]))

            # 처음 정면을 기준으로 저장
            if base_pitch is None:
                base_pitch = pitch

            # 기준 대비 얼마나 턱을 당겼는지 (아래로 숙이면 +)
            pitch_diff = pitch - base_pitch

            color = (0, 255, 0) if abs(pitch_diff) > 10 else (200, 200, 200)
            cv2.putText(frame, f"7.Chin pitch: {pitch_diff:+.0f} deg", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, "(r: reset base angle)", (20, y + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

    cv2.imshow("Gesture Test", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('r'):       # r 누르면 현재 각도를 기준으로 다시 설정
        base_pitch = None

cap.release()
detector.close()
cv2.destroyAllWindows()