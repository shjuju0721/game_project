# ============================================================
#  Face Landmarker 체험용 테스트
#  목표 - FaceMesh 대신 Face Landmarker를 써보고,
#         '블렌드셰이프'(표정 수치)가 어떤 건지 눈으로 확인한다
# ============================================================

import cv2
import mediapipe as mp
from mediapipe.tasks import python              # Face Landmarker는 'tasks' 방식으로 씀
from mediapipe.tasks.python import vision       # 비전(영상) 관련 기능 모음

# ─── 모델 파일 위치 ───
# 이 코드(landmarker_test.py)와 같은 폴더에 face_landmarker.task가 있으므로 파일명만 적음
MODEL_PATH = "face_landmarker.task"

# ─── Face Landmarker 설정 ───
# FaceMesh와 가장 큰 차이: output_face_blendshapes=True 로 켜면
# 'jawOpen(턱 벌림)' 같은 표정 수치 52개를 자동으로 계산해서 줌
base_options = python.BaseOptions(model_asset_path=r"C:\Users\sinhj\embeded\mediapipe_test\face_landmarker.task")
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,        # ★ 핵심: 표정 수치(블렌드셰이프) 켜기
    num_faces=1                          # 얼굴 1명만
)
detector = vision.FaceLandmarker.create_from_options(options)
print("Face Landmarker 준비 완료!")

# ─── 우리가 재활에 관심있는 표정 수치만 골라서 볼 목록 ───
# 52개 다 보면 정신없으니, 입 관련된 것 위주로 추림
WATCH = [
    "jawOpen",          # 턱(입) 벌림  → 우리의 'open'
    "mouthPucker",      # 입 오므리기  → 'suck' 비슷
    "mouthSmileLeft",   # 왼쪽 미소
    "mouthSmileRight",  # 오른쪽 미소  → 'smile'
    "cheekPuff",        # 볼 부풀리기  → 그 어렵던 'puff'! 과연 잡힐까?
]

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)                       # 거울처럼 좌우 반전
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)     # 색 순서 변환

    # Face Landmarker가 요구하는 형식(mp.Image)으로 감싸기
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # 얼굴 인식 실행
    result = detector.detect(mp_image)

    # ─── 블렌드셰이프(표정 수치) 화면에 표시 ───
    if result.face_blendshapes:                      # 얼굴이 잡혔으면
        blendshapes = result.face_blendshapes[0]     # 첫 번째 얼굴의 표정 수치들
        # 이름→점수 형태로 정리 (찾기 쉽게)
        scores = {b.category_name: b.score for b in blendshapes}

        y = 40
        for name in WATCH:                           # 우리가 고른 항목만 표시
            score = scores.get(name, 0)              # 그 표정의 점수 (0~1)
            bar = int(score * 200)                   # 점수를 막대 길이로
            # 글자 표시
            cv2.putText(frame, f"{name}: {score:.2f}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            # 막대 그래프 표시 (점수가 높을수록 길어짐)
            cv2.rectangle(frame, (260, y - 15), (260 + bar, y - 2), (0, 255, 0), -1)
            y += 35

    cv2.imshow("Face Landmarker Test", frame)

    if cv2.waitKey(5) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()