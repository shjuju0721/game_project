# ============================================================
#  STEP 2 : 동작 감지 붙이기 (Face Landmarker 버전)
#  목표 - 왼쪽엔 내 얼굴, 오른쪽 게임 영역에 "지금 무슨 동작 중인지" 글자로 표시
#         입을 벌리면 오른쪽에 "OPEN!" 이라고 뜬다
# ============================================================

import pygame
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ─── 기본 설정 ───
WINDOW_W, WINDOW_H = 1280, 480
HALF_W = WINDOW_W // 2
MODEL_PATH = "C:/Users/sinhj/embeded/game/face_landmarker.task"      # 같은 폴더에 있는 모델 파일

# ─── pygame 준비 ───
pygame.init()
screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
pygame.display.set_caption("꿀떡 꿀떡 게임")
clock = pygame.time.Clock()
# 글자를 쓰려면 폰트가 필요함 (기본 폰트, 크기 지정)
font_big = pygame.font.Font(None, 80)     # 큰 글자 (동작 이름용)
font_small = pygame.font.Font(None, 40)   # 작은 글자 (수치용)

# ─── Face Landmarker 준비 ───
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,        # 표정 수치(블렌드셰이프) 켜기
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

# ─── 색상 ───
COLOR_GAME_BG = (30, 30, 50)
COLOR_WHITE = (255, 255, 255)
COLOR_GREEN = (0, 255, 0)

# ─── 동작 판정 기준값 ───
# jawOpen 수치가 이 값 이상이면 '입 벌림'으로 판정 (실행하며 조절)
OPEN_THRESHOLD = 0.4


def detect_action(blendshapes):
    """표정 수치를 받아서 현재 동작 이름을 반환.
    지금은 '입 벌리기' 하나만. 나중에 동작을 여기 추가하면 됨."""
    # 수치들을 이름→점수 형태로 정리
    scores = {b.category_name: b.score for b in blendshapes}

    jaw_open = scores.get("jawOpen", 0)   # 턱(입) 벌림 정도

    # 판정
    if jaw_open >= OPEN_THRESHOLD:
        return "OPEN!", jaw_open
    else:
        return "...", jaw_open


running = True
while running:

    # ─── (1) 종료 이벤트 ───
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
            running = False

    # ─── (2) 웹캠 읽기 ───
    success, frame = cap.read()
    if not success:
        print("웹캠을 못 읽음!")
        continue
    frame = cv2.flip(frame, 1)

    # ─── (3) Face Landmarker로 얼굴 인식 ───
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    # ─── (4) 동작 판정 ───
    action = "..."          # 기본값 (얼굴 없을 때)
    jaw_value = 0
    if result.face_blendshapes:                  # 얼굴이 잡혔으면
        action, jaw_value = detect_action(result.face_blendshapes[0])

    # ─── (5) 웹캠 영상을 pygame 그림으로 변환 ───
    frame_resized = cv2.resize(rgb, (HALF_W, WINDOW_H))   # rgb는 이미 RGB 색
    frame_surface = pygame.image.frombuffer(
        frame_resized.tobytes(), (HALF_W, WINDOW_H), "RGB"
    )

    # ─── (6) 화면 그리기 ───
    screen.fill(COLOR_GAME_BG)              # 배경 칠하기
    screen.blit(frame_surface, (0, 0))     # 왼쪽에 웹캠

    # 가운데 경계선
    pygame.draw.line(screen, COLOR_WHITE, (HALF_W, 0), (HALF_W, WINDOW_H), 2)

    # ─── 오른쪽 게임 영역에 동작 표시 ───
    # 동작 이름 (큰 글자, 가운데쯤). 입 벌리면 초록색, 아니면 흰색.
    color = COLOR_GREEN if action == "OPEN!" else COLOR_WHITE
    text_surface = font_big.render(action, True, color)
    # 오른쪽 영역의 가운데에 배치 (글자 폭만큼 빼서 중앙 정렬)
    text_x = HALF_W + (HALF_W - text_surface.get_width()) // 2
    screen.blit(text_surface, (text_x, WINDOW_H // 2 - 40))

    # jawOpen 수치도 작게 표시 (기준값 조절에 참고)
    value_text = font_small.render(f"jawOpen: {jaw_value:.2f}", True, COLOR_WHITE)
    screen.blit(value_text, (HALF_W + 30, 30))

    pygame.display.flip()
    clock.tick(30)

# ─── 종료 처리 ───
cap.release()
detector.close()
pygame.quit()