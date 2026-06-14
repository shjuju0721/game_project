# ============================================================
#  STEP 3 : 달리기 게임 (입 벌려서 장애물 점프!)
#  - 떡은 제자리, 장애물이 왼쪽으로 흘러옴
#  - 입 벌리면 점프해서 장애물을 넘는다
#  - 부딪히면 게임오버, 넘으면 점수 +1
# ============================================================

import pygame
import cv2
import random
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ─── 기본 설정 ───
WINDOW_W, WINDOW_H = 1280, 480
HALF_W = WINDOW_W // 2
MODEL_PATH = "C:/Users/sinhj/embeded/game/face_landmarker.task"

# ─── ★ 게임 규칙 (여기 숫자로 난이도 조절) ───
OPEN_THRESHOLD = 0.4      # 입 벌림 판정 기준
OBSTACLE_SPEED = 7        # 장애물이 다가오는 속도 (클수록 어려움)
OBSTACLE_GAP_MIN = 90     # 장애물 사이 최소 간격 (프레임 수)
OBSTACLE_GAP_MAX = 150    # 장애물 사이 최대 간격

# ─── pygame 준비 ───
pygame.init()
screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
pygame.display.set_caption("꿀떡 꿀떡 게임 - 달리기")
clock = pygame.time.Clock()
font_small = pygame.font.Font(None, 36)
font_mid = pygame.font.Font(None, 50)
font_big = pygame.font.Font(None, 90)

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
COLOR_GAME_BG = (135, 180, 220)    # 하늘색 배경
COLOR_WHITE = (255, 255, 255)
COLOR_GROUND = (110, 90, 70)
COLOR_TTEOK = (250, 245, 235)
COLOR_TTEOK_EDGE = (220, 200, 180)
COLOR_CHEEK = (255, 180, 180)
COLOR_OBSTACLE = (90, 160, 90)     # 장애물 색 (초록 선인장 느낌)
COLOR_GREEN = (80, 200, 80)
COLOR_RED = (230, 90, 90)

# ─── 게임 영역 좌표 기준 ───
GAME_LEFT = HALF_W                 # 오른쪽 게임 영역의 왼쪽 끝 x
GROUND_Y = WINDOW_H - 80           # 바닥 높이

# ─── 캐릭터 상태 ───
char_x = GAME_LEFT + 120           # 떡은 게임 영역 왼쪽에 고정
char_y = GROUND_Y
char_vy = 0
GRAVITY = 1.2
JUMP_POWER = -19
is_jumping = False
mouth_was_open = False
CHAR_SIZE = 70                     # 떡 크기 (충돌 판정용)


def draw_tteok(surface, x, y):
    """떡 캐릭터 그리기"""
    body_w, body_h = 90, 70
    body_rect = pygame.Rect(x - body_w//2, y - body_h//2, body_w, body_h)
    pygame.draw.ellipse(surface, COLOR_TTEOK, body_rect)
    pygame.draw.ellipse(surface, COLOR_TTEOK_EDGE, body_rect, 3)
    pygame.draw.circle(surface, COLOR_CHEEK, (x - 25, y + 5), 8)
    pygame.draw.circle(surface, COLOR_CHEEK, (x + 25, y + 5), 8)
    pygame.draw.circle(surface, (50, 50, 50), (x - 18, y - 5), 5)
    pygame.draw.circle(surface, (50, 50, 50), (x + 18, y - 5), 5)
    pygame.draw.ellipse(surface, (180, 120, 120), (x - 8, y + 8, 16, 8))


def reset_game():
    """게임을 처음 상태로 되돌리는 함수 (다시 시작할 때 사용)"""
    return {
        "obstacles": [],       # 장애물 목록 (각 장애물의 x 위치)
        "spawn_timer": 60,     # 다음 장애물 생성까지 남은 시간
        "score": 0,            # 점수
        "game_over": False,    # 게임오버 상태
    }


state = reset_game()   # 게임 상태 시작

running = True
while running:

    # ─── (1) 이벤트 처리 ───
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False
            # 게임오버 상태에서 스페이스바 누르면 다시 시작
            if event.key == pygame.K_SPACE and state["game_over"]:
                state = reset_game()
                char_y = GROUND_Y
                char_vy = 0
                is_jumping = False

    # ─── (2) 웹캠 + 얼굴 인식 ───
    success, frame = cap.read()
    if not success:
        continue
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    jaw_open = 0
    if result.face_blendshapes:
        scores = {b.category_name: b.score for b in result.face_blendshapes[0]}
        jaw_open = scores.get("jawOpen", 0)

    # ─── (3) 점프 판정 (게임오버 아닐 때만) ───
    mouth_open_now = jaw_open >= OPEN_THRESHOLD
    if not state["game_over"]:
        if mouth_open_now and not mouth_was_open and not is_jumping:
            char_vy = JUMP_POWER
            is_jumping = True
    mouth_was_open = mouth_open_now

    # ─── (4) 중력 적용 ───
    char_vy += GRAVITY
    char_y += char_vy
    if char_y >= GROUND_Y:
        char_y = GROUND_Y
        char_vy = 0
        is_jumping = False

    # ─── (5) 장애물 관리 (게임오버 아닐 때만 움직임) ───
    if not state["game_over"]:
        # 새 장애물 생성 타이머
        state["spawn_timer"] -= 1
        if state["spawn_timer"] <= 0:
            # 게임 영역 오른쪽 끝에서 새 장애물 등장
            state["obstacles"].append(WINDOW_W + 20)
            # 다음 생성까지 시간을 랜덤하게 (간격이 일정하지 않게)
            state["spawn_timer"] = random.randint(OBSTACLE_GAP_MIN, OBSTACLE_GAP_MAX)

        # 모든 장애물을 왼쪽으로 이동
        new_obstacles = []
        for ox in state["obstacles"]:
            ox -= OBSTACLE_SPEED        # 왼쪽으로 이동
            if ox > GAME_LEFT - 50:     # 아직 화면 안에 있으면 유지
                new_obstacles.append(ox)
            else:
                # 화면 밖으로 나가면 = 무사히 넘은 것 → 점수 +1
                state["score"] += 1
        state["obstacles"] = new_obstacles

    # ─── (6) 충돌 판정 ───
    # 떡의 네모 영역
    char_rect = pygame.Rect(char_x - CHAR_SIZE//2, char_y - CHAR_SIZE//2,
                            CHAR_SIZE, CHAR_SIZE)
    OBS_W, OBS_H = 40, 60          # 장애물 크기
    for ox in state["obstacles"]:
        # 장애물의 네모 영역 (바닥에 서 있음)
        obs_rect = pygame.Rect(ox, GROUND_Y - OBS_H//2 + 5, OBS_W, OBS_H)
        if char_rect.colliderect(obs_rect):   # 두 네모가 겹치면 = 충돌!
            state["game_over"] = True

    # ─── (7) 웹캠 변환 ───
    frame_resized = cv2.resize(rgb, (HALF_W, WINDOW_H))
    frame_surface = pygame.image.frombuffer(
        frame_resized.tobytes(), (HALF_W, WINDOW_H), "RGB"
    )

    # ─── (8) 화면 그리기 ───
    screen.fill(COLOR_GAME_BG)                          # 하늘색 배경
    screen.blit(frame_surface, (0, 0))                  # 왼쪽 웹캠
    pygame.draw.line(screen, COLOR_WHITE, (HALF_W, 0), (HALF_W, WINDOW_H), 2)

    # 바닥
    pygame.draw.rect(screen, COLOR_GROUND,
                     (HALF_W, GROUND_Y + 35, HALF_W, WINDOW_H - GROUND_Y))

    # 장애물들 그리기
    for ox in state["obstacles"]:
        pygame.draw.rect(screen, COLOR_OBSTACLE,
                         (ox, GROUND_Y - OBS_H//2 + 5, OBS_W, OBS_H))

    # 떡 캐릭터
    draw_tteok(screen, char_x, int(char_y))

    # 점수 표시
    score_text = font_mid.render(f"Score: {state['score']}", True, (40, 40, 60))
    screen.blit(score_text, (HALF_W + 30, 25))

    # 게임오버 화면
    if state["game_over"]:
        over_text = font_big.render("GAME OVER", True, COLOR_RED)
        cx = HALF_W + (HALF_W - over_text.get_width()) // 2
        screen.blit(over_text, (cx, WINDOW_H // 2 - 70))
        # 다시 시작 안내
        retry_text = font_small.render("Press SPACE to retry", True, (40, 40, 60))
        rx = HALF_W + (HALF_W - retry_text.get_width()) // 2
        screen.blit(retry_text, (rx, WINDOW_H // 2 + 20))

    pygame.display.flip()
    clock.tick(30)

# ─── 종료 처리 ───
cap.release()
detector.close()
pygame.quit()