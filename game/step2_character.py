# ============================================================
#  STEP 2 : 떡 캐릭터 + 점프 (Face Landmarker 버전)
#  목표 - 오른쪽에 떡 캐릭터를 그리고, 입을 벌리면 점프시킨다
# ============================================================

import pygame
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ─── 기본 설정 ───
WINDOW_W, WINDOW_H = 1280, 480
HALF_W = WINDOW_W // 2
MODEL_PATH = "C:/Users/sinhj/embeded/game/face_landmarker.task"

# ─── pygame 준비 ───
pygame.init()
screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
pygame.display.set_caption("꿀떡 꿀떡 게임")
clock = pygame.time.Clock()
font_small = pygame.font.Font(None, 40)

# ─── Face Landmarker 준비 ───
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)
cap = cv2.VideoCapture(0)

# ─── 색상 ───
COLOR_GAME_BG = (30, 30, 50)
COLOR_WHITE = (255, 255, 255)
COLOR_GROUND = (80, 70, 60)        # 바닥 색
COLOR_TTEOK = (250, 245, 235)      # 떡 몸통 (흰 떡 느낌)
COLOR_TTEOK_EDGE = (220, 200, 180) # 떡 테두리
COLOR_CHEEK = (255, 180, 180)      # 볼터치 (분홍)

# ─── 동작 판정 기준값 ───
OPEN_THRESHOLD = 0.4

# ============================================================
#  캐릭터(떡)의 상태를 담는 변수들
# ============================================================
# 게임 영역(오른쪽)에서의 좌표. 오른쪽 영역은 x가 HALF_W부터 시작함.
char_x = HALF_W + HALF_W // 2     # 오른쪽 영역의 가로 중앙
GROUND_Y = WINDOW_H - 80          # 바닥 높이 (캐릭터가 서 있을 y 위치)
char_y = GROUND_Y                 # 캐릭터의 현재 세로 위치 (처음엔 바닥)

char_vy = 0          # 세로 속도 (vy = velocity y). 위로 솟으면 음수, 떨어지면 양수
GRAVITY = 1.2        # 중력. 매 순간 vy에 더해져서 캐릭터를 아래로 당김
JUMP_POWER = -18     # 점프 힘. 음수 = 위로 솟음 (화면은 위로 갈수록 y가 작아짐)
is_jumping = False   # 지금 점프 중인지 (점프 중엔 또 점프 못 하게)

# 입 벌림을 '한 번의 동작'으로 세기 위한 장치
# (입을 벌린 채 가만히 있어도 계속 점프하지 않게, '닫혔다가 벌어질 때'만 점프)
mouth_was_open = False


def draw_tteok(surface, x, y):
    """떡 캐릭터를 그리는 함수. 도형을 조합해서 둥근 떡 모양을 만듦.
    x, y = 떡의 중심 위치"""
    # 몸통 (둥근 사각형 느낌 = 타원으로)
    body_w, body_h = 90, 70        # 떡 가로, 세로
    body_rect = pygame.Rect(x - body_w//2, y - body_h//2, body_w, body_h)
    pygame.draw.ellipse(surface, COLOR_TTEOK, body_rect)          # 떡 몸통
    pygame.draw.ellipse(surface, COLOR_TTEOK_EDGE, body_rect, 3)  # 테두리

    # 볼터치 (분홍 동그라미 2개)
    pygame.draw.circle(surface, COLOR_CHEEK, (x - 25, y + 5), 8)
    pygame.draw.circle(surface, COLOR_CHEEK, (x + 25, y + 5), 8)

    # 눈 (까만 점 2개)
    pygame.draw.circle(surface, (50, 50, 50), (x - 18, y - 5), 5)
    pygame.draw.circle(surface, (50, 50, 50), (x + 18, y - 5), 5)

    # 입 (작은 호 = 웃는 입). 간단히 작은 타원으로
    pygame.draw.ellipse(surface, (180, 120, 120), (x - 8, y + 8, 16, 8))


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
        continue
    frame = cv2.flip(frame, 1)

    # ─── (3) 얼굴 인식 + 입 벌림 수치 ───
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    jaw_open = 0
    if result.face_blendshapes:
        scores = {b.category_name: b.score for b in result.face_blendshapes[0]}
        jaw_open = scores.get("jawOpen", 0)

    # ─── (4) 점프 판정 ───
    # 핵심: '닫혀 있다가 벌어지는 순간'에만 점프. (벌린 채 유지하면 연속 점프 X)
    mouth_open_now = jaw_open >= OPEN_THRESHOLD
    if mouth_open_now and not mouth_was_open and not is_jumping:
        # 방금 입을 벌렸고, 점프 중이 아니면 → 점프 시작!
        char_vy = JUMP_POWER     # 위로 솟는 힘을 줌
        is_jumping = True
    mouth_was_open = mouth_open_now   # 이번 상태를 기억 (다음 프레임 비교용)

    # ─── (5) 중력 적용 (캐릭터 위치 업데이트) ───
    char_vy += GRAVITY          # 매 순간 중력을 속도에 더함 (점점 아래로 당겨짐)
    char_y += char_vy           # 속도만큼 위치를 옮김

    # 바닥에 닿으면 멈춤
    if char_y >= GROUND_Y:
        char_y = GROUND_Y       # 바닥보다 아래로 안 내려가게 고정
        char_vy = 0             # 속도 0
        is_jumping = False      # 다시 점프 가능

    # ─── (6) 웹캠 영상 변환 ───
    frame_resized = cv2.resize(rgb, (HALF_W, WINDOW_H))
    frame_surface = pygame.image.frombuffer(
        frame_resized.tobytes(), (HALF_W, WINDOW_H), "RGB"
    )

    # ─── (7) 화면 그리기 ───
    screen.fill(COLOR_GAME_BG)
    screen.blit(frame_surface, (0, 0))                       # 왼쪽 웹캠
    pygame.draw.line(screen, COLOR_WHITE, (HALF_W, 0), (HALF_W, WINDOW_H), 2)  # 경계선

    # 바닥 그리기 (오른쪽 영역에)
    pygame.draw.rect(screen, COLOR_GROUND,
                     (HALF_W, GROUND_Y + 35, HALF_W, WINDOW_H - GROUND_Y))

    # 떡 캐릭터 그리기 (현재 위치에)
    draw_tteok(screen, char_x, int(char_y))

    # jawOpen 수치 표시 (참고용)
    value_text = font_small.render(f"jawOpen: {jaw_open:.2f}", True, COLOR_WHITE)
    screen.blit(value_text, (HALF_W + 30, 30))

    pygame.display.flip()
    clock.tick(30)

# ─── 종료 처리 ───
cap.release()
detector.close()
pygame.quit()