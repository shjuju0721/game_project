import cv2
import mediapipe as mp
import os

mp_face_mesh = mp.solutions.face_mesh

# ─── 설정: 실행할 때마다 사람 이름과 동작을 정함 ───
# 한 번 실행에 '한 사람 + 한 동작'씩 모으는 방식 (제일 헷갈리지 않음)
PERSON = "hyunju"     # ← 찍는 사람 이름 (영어로. 사람 바뀌면 이 줄만 수정)
ACTION = "neutral"       # ← 이번에 모을 동작: "puff" 또는 "neutral"

SAVE_DIR = os.path.join("dataset", ACTION)  # 저장 위치: dataset/puff 또는 dataset/neutral
os.makedirs(SAVE_DIR, exist_ok=True)        # 폴더 없으면 자동 생성

# 이미 저장된 같은 사람 사진 개수를 세서 이어붙이기 (덮어쓰기 방지)
existing = [f for f in os.listdir(SAVE_DIR) if f.startswith(PERSON)]
count = len(existing)
print(f"[{PERSON}] {ACTION} 시작. 현재 {count}장 있음.")
print("스페이스바 = 한 장 저장 / 'a' 누르고 있으면 연속 저장 / 'q' = 종료")

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

        # 얼굴이 화면에 있는지 확인용으로만 검출 (사진엔 점 안 그림)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        face_found = bool(results.multi_face_landmarks)

        # 화면에 안내문 표시 (저장되는 사진엔 안 들어가도록 복사본에만 그림)
        display = frame.copy()
        status = "FACE OK" if face_found else "NO FACE"
        color = (0, 255, 0) if face_found else (0, 0, 255)
        cv2.putText(display, f"{PERSON} / {ACTION} / saved: {count}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display, status, (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imshow("Image Collection", cv2.flip(display, 1))

        key = cv2.waitKey(5) & 0xFF

        # 스페이스바(32) 또는 'a' 키를 누르면 저장 (얼굴이 있을 때만)
        if (key == 32 or key == ord('a')) and face_found:
            count += 1
            filename = f"{PERSON}_{count:03d}.jpg"   # 예: hyunju_001.jpg
            filepath = os.path.join(SAVE_DIR, filename)
            cv2.imwrite(filepath, frame)             # 원본 프레임 저장 (점 없는 깨끗한 사진)

        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print(f"종료. [{PERSON}] {ACTION} 총 {count}장 저장됨.")