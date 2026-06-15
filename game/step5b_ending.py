# ============================================================
#  STEP 5-C : 사람이 장애물처럼 등장하는 엔딩
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
OBSTACLE_SPEED = 9
OBSTACLE_GAP_MIN = 100
OBSTACLE_GAP_MAX = 170
GOAL_SCORE = 5

# ─── pygame 준비 ───
pygame.init()
screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
pygame.display.set_caption("꿀떡 꿀떡 게임")
clock = pygame.time.Clock()
font_small = pygame.font.Font(None, 36)
font_mid = pygame.font.Font(None, 50)
font_big = pygame.font.Font(None, 90)
font_huge = pygame.font.Font(None, 130)

# ─── 입 벌린 사람 이미지 ───
person_img = pygame.image.load("C:/Users/sinhj/embeded/game/ending_image.png").convert_alpha()
PERSON_W, PERSON_H = 520, 500      # 사람 이미지 크기 (더 크게)      # 사람 이미지 크기
person_img = pygame.transform.scale(person_img, (PERSON_W, PERSON_H))

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
COLOR_SUCCESS = (255, 140, 90)

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
STAIR_STEP_W = 90
STAIR_STEP_H = 75

# ─── 사람(엔딩) 관련 위치 ───
PERSON_STOP_X = WINDOW_W - PERSON_W + 40   # 사람이 멈출 x 위치 (오른쪽 끝)
PERSON_Y = GROUND_Y - PERSON_H + 90        # 사람 세로 위치 (바닥에 서게)
# 입 구멍 위치 (사람 이미지 기준 상대 위치 → 멈춘 뒤 절대 위치로 계산)
MOUTH_OFFSET_X = 75      # 사람 이미지 왼쪽 끝에서 입까지 거리
MOUTH_OFFSET_Y = 235     # 사람 이미지 위에서 입까지 거리

def draw_tteok(surface, x, y, size=1.0):
    bw, bh = int(90 * size), int(70 * size)
    body_rect = pygame.Rect(x - bw//2, y - bh//2, bw, bh)
    pygame.draw.ellipse(surface, COLOR_TTEOK, body_rect)
    pygame.draw.ellipse(surface, COLOR_TTEOK_EDGE, body_rect, max(1, int(3*size)))
    if size > 0.4:
        pygame.draw.circle(surface, COLOR_CHEEK, (int(x - 25*size), int(y + 5*size)), max(1, int(8*size)))
        pygame.draw.circle(surface, COLOR_CHEEK, (int(x + 25*size), int(y + 5*size)), max(1, int(8*size)))
        pygame.draw.circle(surface, (50, 50, 50), (int(x - 18*size), int(y - 5*size)), max(1, int(5*size)))
        pygame.draw.circle(surface, (50, 50, 50), (int(x + 18*size), int(y - 5*size)), max(1, int(5*size)))


def draw_game_world(state):
    """게임 배경(하늘, 바닥, 장애물)을 그리는 함수. play와 ending에서 공용."""
    screen.fill(COLOR_GAME_BG)
    screen.blit(frame_surface, (0, 0))
    pygame.draw.line(screen, COLOR_WHITE, (HALF_W, 0), (HALF_W, WINDOW_H), 2)

    # ★ 여기서부터 게임 영역(오른쪽)에만 그리도록 울타리 치기
    game_area = pygame.Rect(HALF_W, 0, HALF_W, WINDOW_H)
    screen.set_clip(game_area)

    pygame.draw.rect(screen, COLOR_GROUND, (HALF_W, GROUND_Y + 35, HALF_W, WINDOW_H - GROUND_Y))
    for obs in state["obstacles"]:
        if obs["type"] == "hole":
            pygame.draw.rect(screen, COLOR_HOLE, (obs["x"], GROUND_Y + 35, HOLE_W, WINDOW_H - GROUND_Y))
    for obs in state["obstacles"]:
        if obs["type"] == "stick":
            pygame.draw.rect(screen, COLOR_STICK, (obs["x"], GROUND_Y - STICK_H//2 + 5, STICK_W, STICK_H))
        elif obs["type"] == "spike":
            base_y = GROUND_Y + 20
            for i in range(3):
                sx = obs["x"] + i * 16
                pygame.draw.polygon(screen, COLOR_SPIKE, [
                    (sx, base_y), (sx + 16, base_y), (sx + 8, base_y - SPIKE_H)])
        elif obs["type"] == "stair":
            for i in range(3):
                step_x = obs["x"] + i * STAIR_STEP_W
                step_h = (i + 1) * STAIR_STEP_H
                step_top = GROUND_Y - step_h
                pygame.draw.rect(screen, COLOR_STICK, (step_x, step_top, STAIR_STEP_W, step_h + 40))
                pygame.draw.rect(screen, COLOR_TTEOK_EDGE, (step_x, step_top, STAIR_STEP_W, step_h + 40), 2)

    # ★ 울타리 풀기 (이 다음부터는 화면 전체에 그릴 수 있게)
    screen.set_clip(None)


def reset_game():
    return {
        "obstacles": [], "spawn_timer": 70, "score": 0,
        "game_over": False,
        "phase": "play",
        "person_x": WINDOW_W + 50,   # 사람이 화면 밖 오른쪽에서 시작
        "ending_timer": 0,
        "show_success": False,
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
            if event.key == pygame.K_SPACE and (state["game_over"] or state["show_success"]):
                state = reset_game()
                char_x = GAME_LEFT + 120
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
    mouth_open_now = jaw_open >= OPEN_THRESHOLD

    frame_resized = cv2.resize(rgb, (HALF_W, WINDOW_H))
    frame_surface = pygame.image.frombuffer(frame_resized.tobytes(), (HALF_W, WINDOW_H), "RGB")

    # ====================================================
    #  PHASE: play
    # ====================================================
    if state["phase"] == "play":
        if not state["game_over"]:
            if mouth_open_now and not mouth_was_open and not is_jumping:
                char_vy = JUMP_POWER
                is_jumping = True

        blocked = False
        char_right = char_x + CHAR_SIZE // 2
        char_bottom = char_y + CHAR_SIZE // 2
        for obs in state["obstacles"]:
            if obs["type"] == "stair":
                for i in range(3):
                    step_x = obs["x"] + i * STAIR_STEP_W
                    step_top = GROUND_Y - (i + 1) * STAIR_STEP_H
                    if (step_x - 6 <= char_right <= step_x + 12 and char_bottom > step_top + 8):
                        blocked = True

        if not state["game_over"] and not blocked:
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
                    if state["score"] >= GOAL_SCORE:
                        state["phase"] = "ending"      # 엔딩 시작!
            state["obstacles"] = new_obstacles

        char_vy += GRAVITY
        char_y += char_vy
        current_ground = GROUND_Y
        over_hole = False
        for obs in state["obstacles"]:
            if obs["type"] == "hole":
                if obs["x"] < char_x < obs["x"] + HOLE_W:
                    over_hole = True
            if obs["type"] == "stair":
                for i in range(3):
                    step_x = obs["x"] + i * STAIR_STEP_W
                    step_top = GROUND_Y - (i + 1) * STAIR_STEP_H
                    if step_x - 10 < char_x < step_x + STAIR_STEP_W + 10:
                        if step_top < current_ground:
                            current_ground = step_top
        falling_into_hole = over_hole and char_y > GROUND_Y and current_ground == GROUND_Y
        if falling_into_hole:
            if char_y >= WINDOW_H + 50:
                state["game_over"] = True
        else:
            if char_y >= current_ground and not over_hole:
                char_y = current_ground
                char_vy = 0
                is_jumping = False
            elif char_y >= current_ground and over_hole and current_ground < GROUND_Y:
                char_y = current_ground
                char_vy = 0
                is_jumping = False

        char_rect = pygame.Rect(char_x - CHAR_SIZE//2, char_y - CHAR_SIZE//2, CHAR_SIZE, CHAR_SIZE)
        for obs in state["obstacles"]:
            if obs["type"] == "stick":
                obs_rect = pygame.Rect(obs["x"], GROUND_Y - STICK_H//2 + 5, STICK_W, STICK_H)
                if char_rect.colliderect(obs_rect):
                    state["game_over"] = True
            elif obs["type"] == "spike":
                obs_rect = pygame.Rect(obs["x"], GROUND_Y - SPIKE_H//2 + 15, SPIKE_W, SPIKE_H)
                if char_rect.colliderect(obs_rect):
                    state["game_over"] = True

        draw_game_world(state)
        draw_tteok(screen, char_x, int(char_y))
        score_text = font_mid.render(f"Score: {state['score']} / {GOAL_SCORE}", True, (40, 40, 60))
        screen.blit(score_text, (HALF_W + 30, 25))
        if state["game_over"]:
            over_text = font_big.render("GAME OVER", True, COLOR_RED)
            cx = HALF_W + (HALF_W - over_text.get_width()) // 2
            screen.blit(over_text, (cx, WINDOW_H // 2 - 70))
            retry_text = font_small.render("Press SPACE to retry", True, (40, 40, 60))
            rx = HALF_W + (HALF_W - retry_text.get_width()) // 2
            screen.blit(retry_text, (rx, WINDOW_H // 2 + 20))

    # ====================================================
    #  PHASE: ending (게임 화면 위에서 사람 등장)
    # ====================================================
    elif state["phase"] == "ending":
        state["ending_timer"] += 1

        # 1) 남은 장애물 계속 흐르게 (사람 등장 전 정리)
        new_obstacles = []
        for obs in state["obstacles"]:
            obs["x"] -= OBSTACLE_SPEED
            if obs["x"] > GAME_LEFT - 300:
                new_obstacles.append(obs)
        state["obstacles"] = new_obstacles

        # 2) 사람이 오른쪽에서 슥 들어와 멈춤
        if state["person_x"] > PERSON_STOP_X:
            state["person_x"] -= OBSTACLE_SPEED
            if state["person_x"] < PERSON_STOP_X:
                state["person_x"] = PERSON_STOP_X

        person_stopped = state["person_x"] <= PERSON_STOP_X

        # 입 구멍의 절대 위치 (사람이 멈춘 곳 기준)
        mouth_x = state["person_x"] + MOUTH_OFFSET_X
        mouth_y = PERSON_Y + MOUTH_OFFSET_Y

        # 3) 사람이 멈추면 떡이 입으로 다가가서 들어감
        tteok_drawn = True
        if person_stopped:
            # 떡을 입 쪽으로 부드럽게 이동
            char_x += (mouth_x - char_x) * 0.08
            char_y += (mouth_y - char_y) * 0.08
            # 입에 충분히 가까워지면 빨려 들어간 것 → 성공
            if abs(char_x - mouth_x) < 30 and abs(char_y - mouth_y) < 30:
                state["show_success"] = True

        # ─── 그리기 ───
        draw_game_world(state)
        # 사람 그리기
        screen.blit(person_img, (int(state["person_x"]), PERSON_Y))
        # 떡 그리기 (크기 그대로 유지하다가 입에 닿으면 사라짐)
        if not state["show_success"]:
            draw_tteok(screen, int(char_x), int(char_y), 1.0)

        if state["show_success"]:
            success_text = font_huge.render("Success!!", True, COLOR_SUCCESS)
            screen.blit(success_text, (80, 100))
            stage_text = font_mid.render("Stage 1 Clear!", True, (60, 60, 80))
            screen.blit(stage_text, (80, 230))
            retry_text = font_small.render("Press SPACE to play again", True, (60, 60, 80))
            screen.blit(retry_text, (80, WINDOW_H - 60))

    mouth_was_open = mouth_open_now

    pygame.display.flip()
    clock.tick(30)

cap.release()
detector.close()
pygame.quit()