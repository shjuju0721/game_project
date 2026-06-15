# ============================================================
#  혀 내밀기(tongueOut) 테스트
#  - 혀를 내밀었을 때 tongueOut 점수가 올라가는지 확인
# ============================================================

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "C:/Users/sinhj/embeded/game/face_landmarker.task"

# Face Landmarker 준비
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

print("=" * 50)
print("혀를 내밀어 보세요. tongueOut 점수를 확인합니다.")
print("종료하려면 q 를 누르세요.")
print("=" * 50)

# tongueOut이 블렌드셰이프 목록에 있는지 한 번만 확인용
checked_list = False

while True:
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    tongue_score = None
    if result.face_blendshapes:
        scores = {b.category_name: b.score for b in result.face_blendshapes[0]}

        # 처음 한 번, 전체 블렌드셰이프 목록에 tongueOut이 있는지 출력
        if not checked_list:
            print("\n[블렌드셰이프 목록 확인]")
            if "tongueOut" in scores:
                print("✅ tongueOut 항목이 있습니다! 인식 가능성 있음")
            else:
                print("❌ tongueOut 항목이 없습니다. 이 모델은 혀를 출력하지 않음")
                print("   (사용 가능한 항목 일부:", list(scores.keys())[:10], "...)")
            print()
            checked_list = True

        tongue_score = scores.get("tongueOut", None)

    # 화면에 점수 표시
    if tongue_score is not None:
        text = f"tongueOut: {tongue_score:.3f}"
        # 점수가 높으면(혀 내민 것으로 판단) 초록, 낮으면 빨강
        color = (0, 255, 0) if tongue_score > 0.3 else (0, 0, 255)
        cv2.putText(frame, text, (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        if tongue_score > 0.3:
            cv2.putText(frame, "TONGUE OUT!", (30, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    else:
        cv2.putText(frame, "tongueOut: N/A (no data)", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    cv2.imshow("Tongue Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
detector.close()
cv2.destroyAllWindows()