import cv2
import mediapipe as mp
import numpy as np
import csv
import os

mp_face_mesh = mp.solutions.face_mesh

# ─── 수집할 동작 목록 ───
# 키보드 숫자키와 동작을 짝지음. 해당 숫자를 누르고 있는 동안 그 동작으로 저장됨.
ACTIONS = {
    ord('0'): "neutral",  # 0번 키 = 중립(무표정)
    ord('1'): "open",     # 1번 키 = 입 벌리기
    ord('2'): "puff",     # 2번 키 = 볼 부풀리기
    ord('3'): "suck",     # 3번 키 = 볼 홀쭉하게
}

CSV_FILE = "face_data.csv"  # 좌표가 쌓일 파일


def extract_features(landmarks):
    """468개 랜드마크를 학습용 숫자 리스트로 변환.
    핵심: 좌표를 '그대로' 쓰면 얼굴 위치/크기에 따라 값이 달라져 학습이 안 됨.
    그래서 코끝(1번)을 기준으로 상대 좌표화하고, 얼굴 폭으로 나눠 정규화함."""
    # 기준점: 코끝
    nose = np.array([landmarks[1].x, landmarks[1].y])
    # 크기 기준자: 얼굴 좌우 폭 (234번 ↔ 454번)
    face_left = np.array([landmarks[234].x, landmarks[234].y])
    face_right = np.array([landmarks[454].x, landmarks[454].y])
    face_width = np.linalg.norm(face_left - face_right)
    if face_width == 0:
        face_width = 1e-6  # 0으로 나누기 방지

    features = []
    for lm in landmarks:
        # (각 점 - 코끝) / 얼굴폭  →  위치·크기에 무관한 값
        rel_x = (lm.x - nose[0]) / face_width
        rel_y = (lm.y - nose[1]) / face_width
        features.extend([rel_x, rel_y])
    return features  # 468점 × (x,y) = 936개 숫자


# CSV 파일이 없으면 헤더(열 이름) 먼저 작성
if not os.path.exists(CSV_FILE):
    header = ["label"]  # 첫 열은 정답(동작 이름)
    for i in range(468):
        header += [f"x{i}", f"y{i}"]
    with open(CSV_FILE, "w", newline="") as f:
        csv.writer(f).writerow(header)

cap = cv2.VideoCapture(0)

# 동작별로 몇 개씩 모았는지 세는 카운터
counts = {name: 0 for name in ACTIONS.values()}

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

        # 화면 안내문
        y = 30
        cv2.putText(image, "Hold key to record:  0=neutral 1=open 2=puff 3=suck",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 30
        for name, c in counts.items():
            cv2.putText(image, f"{name}: {c}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y += 25

        key = cv2.waitKey(5) & 0xFF

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            # 동작 키를 누르고 있으면 그 프레임의 좌표를 저장
            if key in ACTIONS:
                label = ACTIONS[key]
                features = extract_features(landmarks)
                with open(CSV_FILE, "a", newline="") as f:
                    csv.writer(f).writerow([label] + features)
                counts[label] += 1

        cv2.imshow("Data Collection", cv2.flip(image, 1))

        if key == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
print("수집 종료. 동작별 개수:", counts)