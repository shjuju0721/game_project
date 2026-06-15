# ============================================================
#  볼 부풀리기 판정 - 볼 아래쪽(B,A,C) + 입술 오므림 조합
#  사용법: 가만히 있을 때 'r' 눌러 기준 등록 → 볼 부풀려보기
# ============================================================

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "C:/Users/sinhj/embeded/game/face_landmarker.task"

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)
cap = cv2.VideoCapture(0)

print("=" * 50)
print("1) 가만히 있는 상태에서 'r' 키로 기준 등록")
print("2) 볼을 부풀려서 CHEEK PUFF! 가 뜨는지 확인")
print("종료: q")
print("=" * 50)


def dist(p1, p2):
    return ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) ** 0.5


base_cheek = None       # 평소 볼 너비 기준
base_pucker = None      # 평소 입술 기준

while True:
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    if result.face_landmarks and result.face_blendshapes:
        lm = result.face_landmarks[0]
        scores = {b.category_name: b.score for b in result.face_blendshapes[0]}

        face_height = dist(lm[10], lm[152])
        # 볼 아래쪽 신호 = A, B, C 평균 (잘 변한 것들)
        a = dist(lm[61], lm[135]) / face_height
        b = dist(lm[291], lm[364]) / face_height
        c = dist(lm[137], lm[366]) / face_height
        cheek = (a + b + c) / 3

        pucker = scores.get("mouthPucker", 0)

        # 기준 대비 변화량
        if base_cheek is not None:
            cheek_diff = cheek - base_cheek          # 볼이 얼마나 넓어졌나
            pucker_diff = pucker - base_pucker        # 입술이 얼마나 오므려졌나

            cv2.putText(frame, f"cheek +{cheek_diff:.3f}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2)
            cv2.putText(frame, f"pucker +{pucker_diff:.2f}", (20, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2)

            # 판정: 볼도 넓어지고 입술도 오므려지면
            is_puff = cheek_diff > 0.020 and pucker_diff > 0.07
            color = (0, 255, 0) if is_puff else (0, 0, 255)
            cv2.putText(frame, "CHEEK PUFF!" if is_puff else "...", (20, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
        else:
            cv2.putText(frame, "Press 'r' to set base", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

        # r 누르면 현재를 기준으로
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('r'):
            base_cheek = cheek
            base_pucker = pucker
            print(f"기준 등록! cheek={cheek:.3f}, pucker={pucker:.2f}")
    else:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.imshow("Cheek Puff Test", frame)

cap.release()
detector.close()
cv2.destroyAllWindows()