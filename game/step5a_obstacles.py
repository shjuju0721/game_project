# ============================================================
#  STEP 5-A : 장애물 4종류 (가시 / 막대기 / 땅구멍 / 계단)
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

# ─── ★ 게임 규칙 ───
OPEN_THRESHOLD = 0.4
OBSTACLE_SPEED = 9            # 속도 (7 → 9로 높임)
OBSTACLE_GAP_MIN = 100
OBSTACLE_GAP_MAX = 170

# ─── pygame 준비 ───
pygame.init()
screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
pygame.display.set_caption("꿀떡 꿀떡 게임 - 달리기")
clock = pygame.time.Clock()
font_small = pygame.font.Font(None, 36)
font_mid = pygame.font.Font(None, 50)
font_big = pygame.font.Font(None, 90)

# ─── Face Landmarker ───
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options, output_face_blendshapes=True, num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)
cap = cv2.VideoCapture(0)

# ─── 색상 ───
COLOR_GAME_BG = (135, 180, 220)
COLOR_WHITE = (255, 255, 255)
COLOR_GROUND = (110, 90, 70)
COLOR_TTEOK = (250, 245, 235)
COLOR_TTEOK_EDGE = (220, 200, 180)
COLOR_CHEEK = (255, 180, 180)
COLOR_STICK = (90, 160, 90)
COLOR_SPIKE = (200, 80, 80)
COLOR_HOLE = (40, 40, 60)
COLOR_RED = (230, 90, 90)

# ─── 좌표 기준 ───
GAME_LEFT = HALF_W
GROUND_Y = WINDOW_H - 80

# ─── 캐릭터 상태 ───
char_x = GAME_LEFT + 120
char_y = GROUND_Y
char_vy = 0
GRAVITY = 1.2
JUMP_POWER = -21
is_jumping = False
mouth_was_open = False
CHAR_SIZE = 70

# ─── 장애물 크기 ───
STICK_W, STICK_H = 40, 60
SPIKE_W, SPIKE_H = 50, 45
HOLE_W = 140
STAIR_STEP_W = 90            # 계단 한 칸 너비
STAIR_STEP_H = 75            # 계단 한 칸 높이 (60 → 75, 높이차 키움)


def draw_tteok(surface, x, y):
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
    return {
        "obstacles": [],
        "spawn_timer": 70,
        "score": 0,
        "game_over": False,
    }


state = reset_game()

running = True
while running:

    # ─── (1) 이벤트 ───
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False
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

    # ─── (3) 점프 판정 ───
    mouth_open_now = jaw_open >= OPEN_THRESHOLD
    if not state["game_over"]:
        if mouth_open_now and not mouth_was_open and not is_jumping:
            char_vy = JUMP_POWER
            is_jumping = True
    mouth_was_open = mouth_open_now

   # ─── (4) 장애물 생성 & 이동 ───
    # '떡이 다음 계단 턱에 막혀 있는지' 판단
    blocked = False
    char_right = char_x + CHAR_SIZE // 2     # 떡의 오른쪽 끝
    char_bottom = char_y + CHAR_SIZE // 2    # 떡의 아래 끝
    for obs in state["obstacles"]:
        if obs["type"] == "stair":
            for i in range(3):
                step_x = obs["x"] + i * STAIR_STEP_W          # i번째 칸 왼쪽(턱)
                step_top = GROUND_Y - (i + 1) * STAIR_STEP_H   # i번째 칸 윗면
                # 떡 오른쪽이 이 턱 바로 앞에 닿았는데, 떡이 이 칸 윗면보다 아래면 → 막힘
                if (step_x - 6 <= char_right <= step_x + 12
                        and char_bottom > step_top + 8):
                    blocked = True

    if not state["game_over"]:
        # 막혀 있지 않을 때만 장애물 생성 & 이동
        if not blocked:
            state["spawn_timer"] -= 1
            if state["spawn_timer"] <= 0:
                obs_type = random.choice(["stick", "spike", "hole", "stair"])
                state["obstacles"].append({"x": WINDOW_W + 20, "type": obs_type})
                state["spawn_timer"] = random.randint(OBSTACLE_GAP_MIN, OBSTACLE_GAP_MAX)

            new_obstacles = []
            for obs in state["obstacles"]:
                obs["x"] -= OBSTACLE_SPEED
                if obs["x"] > GAME_LEFT - 300:
                    new_obstacles.append(obs)
                else:
                    state["score"] += 1
            state["obstacles"] = new_obstacles

    # ─── (5) 중력 적용 ───
    char_vy += GRAVITY
    char_y += char_vy

    # ─── (6) 바닥 판정 (구멍 / 계단 고려) ───
    current_ground = GROUND_Y
    over_hole = False
    char_bottom = char_y + CHAR_SIZE // 2

    for obs in state["obstacles"]:
        if obs["type"] == "hole":
            if obs["x"] < char_x < obs["x"] + HOLE_W:
                over_hole = True
        if obs["type"] == "stair":
            for i in range(3):
                step_x = obs["x"] + i * STAIR_STEP_W
                step_top = GROUND_Y - (i + 1) * STAIR_STEP_H
                # 떡 중심이 이 칸의 좌우 범위 안에 있으면, 이 칸 윗면을 착지 후보로
                if step_x - 10 < char_x < step_x + STAIR_STEP_W + 10:
                    if step_top < current_ground:
                        current_ground = step_top

    # 한 번 구멍에 빠지기 시작했는지 기억하는 표시
    # (떨어지는 중이고 + 발밑이 땅보다 아래로 내려갔으면 = 이미 빠지는 중)
    falling_into_hole = over_hole and char_y > GROUND_Y and current_ground == GROUND_Y

    if falling_into_hole:
        # 빠지는 중 → 절대 착지 안 시킴, 계속 떨어뜨림
        if char_y >= WINDOW_H + 50:
            state["game_over"] = True
    else:
        # 정상 상황 (땅 또는 계단 위)
        if char_y >= current_ground and not over_hole:
            char_y = current_ground
            char_vy = 0
            is_jumping = False
        elif char_y >= current_ground and over_hole and current_ground < GROUND_Y:
            # 구멍 위지만 계단도 있는 특수 경우 → 계단에 착지
            char_y = current_ground
            char_vy = 0
            is_jumping = False

    # ─── (7) 충돌 판정 (가시 / 막대기만! 계단·구멍은 여기 없음) ───
    char_rect = pygame.Rect(char_x - CHAR_SIZE//2, char_y - CHAR_SIZE//2,
                            CHAR_SIZE, CHAR_SIZE)
    for obs in state["obstacles"]:
        if obs["type"] == "stick":
            obs_rect = pygame.Rect(obs["x"], GROUND_Y - STICK_H//2 + 5,
                                   STICK_W, STICK_H)
            if char_rect.colliderect(obs_rect):
                state["game_over"] = True
        elif obs["type"] == "spike":
            obs_rect = pygame.Rect(obs["x"], GROUND_Y - SPIKE_H//2 + 15,
                                   SPIKE_W, SPIKE_H)
            if char_rect.colliderect(obs_rect):
                state["game_over"] = True

    # ─── (8) 웹캠 변환 ───
    frame_resized = cv2.resize(rgb, (HALF_W, WINDOW_H))
    frame_surface = pygame.image.frombuffer(
        frame_resized.tobytes(), (HALF_W, WINDOW_H), "RGB"
    )

    # ─── (9) 화면 그리기 ───
    screen.fill(COLOR_GAME_BG)
    screen.blit(frame_surface, (0, 0))
    pygame.draw.line(screen, COLOR_WHITE, (HALF_W, 0), (HALF_W, WINDOW_H), 2)

    # 바닥 (구멍 자리는 나중에 어둡게 덮음)
    pygame.draw.rect(screen, COLOR_GROUND,
                     (HALF_W, GROUND_Y + 35, HALF_W, WINDOW_H - GROUND_Y))
    for obs in state["obstacles"]:
        if obs["type"] == "hole":
            pygame.draw.rect(screen, COLOR_HOLE,
                             (obs["x"], GROUND_Y + 35, HOLE_W, WINDOW_H - GROUND_Y))

    # 장애물 그리기 (가시 / 막대기 / 계단)
    for obs in state["obstacles"]:
        if obs["type"] == "stick":
            pygame.draw.rect(screen, COLOR_STICK,
                             (obs["x"], GROUND_Y - STICK_H//2 + 5, STICK_W, STICK_H))
        elif obs["type"] == "spike":
            base_y = GROUND_Y + 20
            for i in range(3):
                sx = obs["x"] + i * 16
                pygame.draw.polygon(screen, COLOR_SPIKE, [
                    (sx, base_y),
                    (sx + 16, base_y),
                    (sx + 8, base_y - SPIKE_H),
                ])
        elif obs["type"] == "stair":
            # 계단 3칸을 점점 높게 그림
            for i in range(3):
                step_x = obs["x"] + i * STAIR_STEP_W
                step_h = (i + 1) * STAIR_STEP_H
                step_top = GROUND_Y - step_h
                pygame.draw.rect(screen, COLOR_STICK,
                                 (step_x, step_top, STAIR_STEP_W, step_h + 40))
                pygame.draw.rect(screen, COLOR_TTEOK_EDGE,
                                 (step_x, step_top, STAIR_STEP_W, step_h + 40), 2)

    # 떡 캐릭터
    draw_tteok(screen, char_x, int(char_y))

    # 점수
    score_text = font_mid.render(f"Score: {state['score']}", True, (40, 40, 60))
    screen.blit(score_text, (HALF_W + 30, 25))

    # 게임오버
    if state["game_over"]:
        over_text = font_big.render("GAME OVER", True, COLOR_RED)
        cx = HALF_W + (HALF_W - over_text.get_width()) // 2
        screen.blit(over_text, (cx, WINDOW_H // 2 - 70))
        retry_text = font_small.render("Press SPACE to retry", True, (40, 40, 60))
        rx = HALF_W + (HALF_W - retry_text.get_width()) // 2
        screen.blit(retry_text, (rx, WINDOW_H // 2 + 20))

    pygame.display.flip()
    clock.tick(30)

cap.release()
detector.close()
pygame.quit()