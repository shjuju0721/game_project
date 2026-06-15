# ============================================================
#  볼 부풀리기 측정 테스트 (FaceMesh 가로 길이 방식)
#  - 여러 거리 비율을 동시에 보면서 뭐가 잘 반응하는지 찾기
# ============================================================

import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

print("=" * 50)
print("정면을 본 뒤, 볼을 부풀려 보세요.")
print("어떤 값이 가장 크게 변하는지 확인하세요.")
print("종료: q")
print("=" * 50)


def dist(p1, p2):
    """두 점 사이 거리"""
    return ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) ** 0.5


while True:
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        lm = result.multi_face_landmarks[0].landmark

        # ─── 기준: 얼굴 너비 (왼쪽 끝 234 ↔ 오른쪽 끝 454) ───
        face_width = dist(lm[234], lm[454])

        # ─── 측정 1: 전체 얼굴 너비 (볼 바깥점) ───
        # 볼 부위 윤곽선 점들: 왼쪽 볼 50, 오른쪽 볼 280
        cheek_width = dist(lm[50], lm[280])
        ratio1 = cheek_width / face_width

        # ─── 측정 2: 왼쪽 볼 (코 옆 점 ↔ 볼 바깥점) ───
        # 코 옆 왼쪽 129, 왼쪽 볼 바깥 234
        left_cheek = dist(lm[129], lm[234])
        ratio2 = left_cheek / face_width

        # ─── 측정 3: 오른쪽 볼 (코 옆 점 ↔ 볼 바깥점) ───
        # 코 옆 오른쪽 358, 오른쪽 볼 바깥 454
        right_cheek = dist(lm[358], lm[454])
        ratio3 = right_cheek / face_width

        # ─── 측정 4: 양 볼 합 (입꼬리 기준 볼 너비) ───
        # 왼쪽 볼 점 207 ↔ 오른쪽 볼 점 427
        mouth_cheek = dist(lm[207], lm[427])
        ratio4 = mouth_cheek / face_width

        # 화면에 표시
        values = {
            "1. cheek(50-280)": ratio1,
            "2. L cheek(129-234)": ratio2,
            "3. R cheek(358-454)": ratio3,
            "4. mouth(207-427)": ratio4,
        }
        y = 50
        for name, val in values.items():
            cv2.putText(frame, f"{name}: {val:.4f}", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y += 45

    cv2.imshow("Cheek Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()