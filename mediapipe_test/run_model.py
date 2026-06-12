import cv2
import mediapipe as mp
import numpy as np
import joblib   # 저장한 모델 불러오기

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

MODEL_FILE = "face_model.pkl"
model = joblib.load(MODEL_FILE)   # 학습된 모델 불러오기
print("모델 불러오기 완료!")


def extract_features(landmarks):
    """⚠️ 수집 코드와 100% 똑같아야 함. 모델이 이 형식으로 배웠기 때문."""
    nose = np.array([landmarks[1].x, landmarks[1].y])
    face_left = np.array([landmarks[234].x, landmarks[234].y])
    face_right = np.array([landmarks[454].x, landmarks[454].y])
    face_width = np.linalg.norm(face_left - face_right)
    if face_width == 0:
        face_width = 1e-6
    features = []
    for lm in landmarks:
        rel_x = (lm.x - nose[0]) / face_width
        rel_y = (lm.y - nose[1]) / face_width
        features.extend([rel_x, rel_y])
    return features


# 동작 이름을 한글로 보기 좋게 (선택)
LABEL_KR = {
    "neutral": "Neutral",
    "open": "Open Mouth",
    "puff": "Puff",
    "suck": "Suck",
}

cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame.flags.writeable = False
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        frame.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            # 1) 좌표 → 숫자 변환 (수집 때와 동일)
            features = extract_features(landmarks)

            # 2) 모델에게 예측 요청 (입력은 2차원 형태여야 해서 [features]로 감쌈)
            prediction = model.predict([features])[0]          # 예측된 동작 이름
            proba = model.predict_proba([features])[0]         # 각 동작일 확률
            confidence = max(proba)                            # 가장 높은 확률 = 확신도

            # 3) 화면에 표시
            text = f"{LABEL_KR.get(prediction, prediction)}  ({confidence:.0%})"
            cv2.putText(image, text, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)

            # 입술 윤곽 그리기 (참고용)
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=results.multi_face_landmarks[0],
                connections=mp_face_mesh.FACEMESH_LIPS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_contours_style()
            )

        cv2.imshow("Real-time Recognition", cv2.flip(image, 1))

        if cv2.waitKey(5) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()