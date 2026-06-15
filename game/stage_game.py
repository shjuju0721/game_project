# ============================================================
#  꿀떡 꿀떡 게임 - 6단계 버전
#  각 단계마다 다른 얼굴 동작으로 꿀떡이를 점프시킨다.
#  순서: 턱당기기 → 입벌리기 → 입다물기 → 웃기 → 입술오므리기 → 볼부풀리기
# ============================================================

# ─── 필요한 도구(라이브러리) 불러오기 ───
import pygame          # 게임 화면을 만드는 도구
import cv2             # 웹캠 영상을 다루는 도구
import random          # 장애물을 무작위로 뽑을 때 사용
import numpy as np     # 숫자 계산(특히 머리 각도 계산)에 사용
import mediapipe as mp # 얼굴 인식 도구
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ─── 화면 크기 설정 ───
WINDOW_W, WINDOW_H = 1280, 480   # 전체 창 가로, 세로 크기
HALF_W = WINDOW_W // 2           # 화면 절반 (왼쪽=웹캠, 오른쪽=게임)
MODEL_PATH = "C:/Users/sinhj/embeded/game/face_landmarker.task"  # 얼굴인식 모델 파일 위치

# ─── 게임 규칙 숫자들 ───
OBSTACLE_SPEED = 9        # 장애물이 왼쪽으로 흐르는 속도
OBSTACLE_GAP_MIN = 90   # 장애물 사이 최소 간격(시간)
OBSTACLE_GAP_MAX = 160   # 장애물 사이 최대 간격(시간)
GOAL_SCORE = 3          # 이 점수를 채우면 그 단계 클리어 (테스트할 땐 3으로 낮춰도 됨)

# ─── 6개 단계 정의 (순서대로) ───
# key  = 프로그램이 동작을 구분하는 이름(영어 코드)
# name = 화면에 보여줄 이름
STAGES = [
    {"key": "chin",   "name": "Chin Tuck"},      # 1단계: 턱 당기기
    {"key": "open",   "name": "Open Mouth"},     # 2단계: 입 벌리기
    {"key": "press",  "name": "Close Mouth"},    # 3단계: 입 다물기
    {"key": "smile",  "name": "Smile"},          # 4단계: 웃기
    {"key": "pucker", "name": "Pucker Lips"},    # 5단계: 입술 오므리기
    {"key": "puff",   "name": "Cheek Puff"},     # 6단계: 볼 부풀리기
]

# ─── pygame(게임 화면) 준비 ───
pygame.init()                                          # pygame 시작
screen = pygame.display.set_mode((WINDOW_W, WINDOW_H)) # 창 만들기
pygame.display.set_caption("꿀떡 꿀떡 게임")            # 창 제목
clock = pygame.time.Clock()                            # 게임 속도(프레임) 조절용
# 글자 크기별 폰트 (None=기본폰트, 숫자=크기)
font_small = pygame.font.Font(None, 36)
font_mid = pygame.font.Font(None, 50)
font_big = pygame.font.Font(None, 90)
font_huge = pygame.font.Font(None, 110)

# ─── 엔딩에 쓸 '입 벌린 사람' 이미지 불러오기 ───
person_img = pygame.image.load("C:/Users/sinhj/embeded/game/ending_image.png").convert_alpha()
PERSON_W, PERSON_H = 520, 500                              # 사람 이미지 크기
person_img = pygame.transform.scale(person_img, (PERSON_W, PERSON_H))  # 그 크기로 조절

# ─── 얼굴 인식기(Face Landmarker) 준비 ───
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,                  # 표정 점수 받기 (웃기, 오므리기 등)
    output_facial_transformation_matrixes=True,    # 머리 각도 받기 (턱 당기기용)
    num_faces=1                                     # 얼굴 1명만 인식
)
detector = vision.FaceLandmarker.create_from_options(options)
cap = cv2.VideoCapture(0)                          # 0번 웹캠 켜기

# ─── 색깔 정의 (R, G, B) ───
COLOR_GAME_BG = (135, 180, 220)   # 게임 배경 하늘색
COLOR_WHITE = (255, 255, 255)     # 흰색
COLOR_GROUND = (110, 90, 70)      # 땅 갈색
COLOR_TTEOK = (250, 245, 235)     # 떡 몸통색
COLOR_TTEOK_EDGE = (220, 200, 180)# 떡 테두리색
COLOR_CHEEK = (255, 180, 180)     # 떡 볼터치색
COLOR_STICK = (90, 160, 90)       # 막대기/계단 초록
COLOR_SPIKE = (200, 80, 80)       # 가시 빨강
COLOR_HOLE = (40, 40, 60)         # 구멍 어두운색
COLOR_RED = (230, 90, 90)         # 게임오버 글자
COLOR_SUCCESS = (255, 140, 90)    # 성공 글자 주황
COLOR_GUIDE = (40, 40, 90)        # 안내 글자 남색

# ─── 게임 좌표 기준 ───
GAME_LEFT = HALF_W            # 게임 영역 왼쪽 끝(화면 가운데)
GROUND_Y = WINDOW_H - 80     # 바닥의 높이(y좌표)

# ─── 캐릭터(꿀떡이) 상태 변수 ───
char_x = GAME_LEFT + 120     # 떡의 가로 위치(고정)
char_y = GROUND_Y            # 떡의 세로 위치(점프하면 변함)
char_vy = 0                  # 떡의 수직 속도(점프/낙하)
GRAVITY = 1.2                # 중력(매 순간 아래로 당기는 힘)
JUMP_POWER = -21             # 점프 힘(음수=위로)
is_jumping = False           # 지금 점프 중인지
CHAR_SIZE = 70               # 떡의 충돌 판정 크기

# ─── 장애물 크기들 ───
STICK_W, STICK_H = 40, 60        # 막대기 가로, 세로
SPIKE_W, SPIKE_H = 50, 45        # 가시 가로, 세로
HOLE_W = 140                     # 구멍 너비
STAIR_STEP_W = 90                # 계단 한 칸 너비
STAIR_STEP_H = 75                # 계단 한 칸 높이

# ─── 엔딩에서 사람/입 위치 ───
PERSON_STOP_X = WINDOW_W - PERSON_W + 40   # 사람이 멈출 가로 위치
PERSON_Y = GROUND_Y - PERSON_H + 90        # 사람의 세로 위치(바닥에 맞춤)
MOUTH_OFFSET_X = 75                        # 사람 이미지 왼쪽에서 입까지 가로 거리
MOUTH_OFFSET_Y = 235                       # 사람 이미지 위에서 입까지 세로 거리

# ─── 동작 판정에 쓸 '기준값'들 ───
# (턱 당기기, 볼 부풀리기는 '평소 상태'와 비교해야 해서 기준이 필요)
base_pitch = None              # 평소 머리 각도(턱 당기기 기준)
base_cheek = None              # 평소 볼 너비(볼 부풀리기 기준)
base_pucker_for_puff = None    # 평소 입술 상태(볼 부풀리기용)


def dist(p1, p2):
    """두 점 사이의 거리를 구하는 함수 (피타고라스)"""
    return ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) ** 0.5


def detect_gesture(key, scores, lm, pitch):
    """
    현재 단계의 동작(key)이 지금 감지되는지 판단해서
    감지되면 True, 아니면 False를 돌려준다.
    scores = 표정 점수들, lm = 얼굴 점들, pitch = 머리 각도
    """
    global base_pitch, base_cheek, base_pucker_for_puff

    if key == "open":      # 입 벌리기
        # jawOpen(입 벌림 정도)이 0.4 넘으면 입 벌린 것
        return scores.get("jawOpen", 0) > 0.4

    elif key == "press":   # 입 다물기 (입술 꾹)
        # 왼쪽+오른쪽 입술 누름 점수의 평균이 0.4 넘으면
        press = (scores.get("mouthPressLeft", 0) + scores.get("mouthPressRight", 0)) / 2
        return press > 0.4

    elif key == "smile":   # 웃기
        # 왼쪽+오른쪽 미소 점수의 평균이 0.4 넘으면
        smile = (scores.get("mouthSmileLeft", 0) + scores.get("mouthSmileRight", 0)) / 2
        return smile > 0.4

    elif key == "pucker":  # 입술 오므리기
        # 평소(0.89)보다 높게 잡아야 함. 오므리면 1.0이 되므로 0.95 기준
        return scores.get("mouthPucker", 0) > 0.95

    elif key == "chin":    # 턱 당기기
        # 기준 각도가 아직 없으면 판정 불가
        if base_pitch is None or pitch is None:
            return False
        # 턱을 당기면 각도가 '마이너스'로 가므로, 변화량의 절댓값으로 판정
        return abs(pitch - base_pitch) > 12

    elif key == "puff":    # 볼 부풀리기 (우리가 직접 만든 판정!)
        if lm is None or base_cheek is None:
            return False
        # 볼 아래쪽 너비 3곳(A,B,C)을 재서 평균
        face_height = dist(lm[10], lm[152])             # 얼굴 세로 길이(기준)
        a = dist(lm[61], lm[135]) / face_height          # 왼쪽 입꼬리-턱선
        b = dist(lm[291], lm[364]) / face_height         # 오른쪽 입꼬리-턱선
        c = dist(lm[137], lm[366]) / face_height         # 양쪽 턱선 너비
        cheek = (a + b + c) / 3
        pucker = scores.get("mouthPucker", 0)            # 입술 오므림 점수
        # 볼도 넓어지고(+0.02) 입술도 오므려지면(+0.07) 볼 부풀린 것
        return (cheek - base_cheek) > 0.020 and (pucker - base_pucker_for_puff) > 0.07

    return False


def draw_tteok(surface, x, y, size=1.0):
    """떡(꿀떡이) 캐릭터를 그리는 함수. size로 크기 조절."""
    bw, bh = int(90 * size), int(70 * size)               # 몸통 가로, 세로
    body_rect = pygame.Rect(x - bw//2, y - bh//2, bw, bh) # 몸통 영역
    pygame.draw.ellipse(surface, COLOR_TTEOK, body_rect)             # 몸통(타원)
    pygame.draw.ellipse(surface, COLOR_TTEOK_EDGE, body_rect, max(1, int(3*size)))  # 테두리
    if size > 0.4:   # 너무 작지 않을 때만 얼굴 그리기
        # 양 볼(분홍 동그라미)
        pygame.draw.circle(surface, COLOR_CHEEK, (int(x-25*size), int(y+5*size)), max(1,int(8*size)))
        pygame.draw.circle(surface, COLOR_CHEEK, (int(x+25*size), int(y+5*size)), max(1,int(8*size)))
        # 양 눈(검은 점)
        pygame.draw.circle(surface, (50,50,50), (int(x-18*size), int(y-5*size)), max(1,int(5*size)))
        pygame.draw.circle(surface, (50,50,50), (int(x+18*size), int(y-5*size)), max(1,int(5*size)))


def draw_game_world(state):
    """게임 배경(웹캠, 하늘, 바닥, 장애물)을 그리는 함수."""
    screen.fill(COLOR_GAME_BG)                  # 하늘색으로 채우기
    screen.blit(frame_surface, (0, 0))          # 왼쪽에 웹캠 영상
    pygame.draw.line(screen, COLOR_WHITE, (HALF_W, 0), (HALF_W, WINDOW_H), 2)  # 가운데 구분선

    # 장애물이 왼쪽(웹캠)으로 넘어가지 않게 게임 영역에만 그리도록 울타리 치기
    game_area = pygame.Rect(HALF_W, 0, HALF_W, WINDOW_H)
    screen.set_clip(game_area)

    # 바닥 그리기
    pygame.draw.rect(screen, COLOR_GROUND, (HALF_W, GROUND_Y+35, HALF_W, WINDOW_H-GROUND_Y))
    # 구멍 먼저 그리기(바닥 위에 어둡게)
    for obs in state["obstacles"]:
        if obs["type"] == "hole":
            pygame.draw.rect(screen, COLOR_HOLE, (obs["x"], GROUND_Y+35, HOLE_W, WINDOW_H-GROUND_Y))
    # 나머지 장애물 그리기
    for obs in state["obstacles"]:
        if obs["type"] == "stick":       # 막대기
            pygame.draw.rect(screen, COLOR_STICK, (obs["x"], GROUND_Y-STICK_H//2+5, STICK_W, STICK_H))
        elif obs["type"] == "spike":     # 가시(삼각형 3개)
            base_y = GROUND_Y + 20
            for i in range(3):
                sx = obs["x"] + i*16
                pygame.draw.polygon(screen, COLOR_SPIKE, [(sx,base_y),(sx+16,base_y),(sx+8,base_y-SPIKE_H)])
        elif obs["type"] == "stair":     # 계단(3칸)
            for i in range(3):
                step_x = obs["x"] + i*STAIR_STEP_W
                step_h = (i+1)*STAIR_STEP_H
                step_top = GROUND_Y - step_h
                pygame.draw.rect(screen, COLOR_STICK, (step_x, step_top, STAIR_STEP_W, step_h+40))
                pygame.draw.rect(screen, COLOR_TTEOK_EDGE, (step_x, step_top, STAIR_STEP_W, step_h+40), 2)

    screen.set_clip(None)   # 울타리 풀기(이제 화면 전체에 그릴 수 있음)


def reset_stage():
    """한 단계를 처음 상태로 되돌리는 함수. 단계 시작/재시작 때 호출."""
    return {
        "obstacles": [],          # 장애물 목록(비움)
        "spawn_timer": 70,        # 다음 장애물까지 시간
        "score": 0,               # 점수
        "game_over": False,       # 게임오버 여부
        "phase": "ready",         # 현재 화면: ready(안내)/play(게임)/ending(엔딩)/cleared(클리어)
        "person_x": WINDOW_W + 50,# 엔딩 사람의 시작 위치(화면 밖 오른쪽)
        "show_success": False,    # 성공 표시 여부
    }


# ─── 게임 시작 준비 ───
stage_index = 0          # 현재 몇 단계인지 (0=1단계)
state = reset_stage()    # 첫 단계 상태 만들기
gesture_was_on = False   # '직전 순간에 동작이 켜져 있었나' (한 번 동작=한 번 점프 위해)

running = True
while running:
    current_stage = STAGES[stage_index]   # 지금 단계 정보 가져오기

    # ─── (1) 키보드/창닫기 이벤트 처리 ───
    for event in pygame.event.get():
        if event.type == pygame.QUIT:     # 창 닫기 버튼
            running = False
        if event.type == pygame.KEYDOWN:  # 키를 눌렀을 때
            if event.key == pygame.K_q:   # q = 종료
                running = False
            # 안내 화면에서 스페이스 → 게임 시작
            if event.key == pygame.K_SPACE and state["phase"] == "ready":
                state["phase"] = "play"
            # 클리어 화면에서 스페이스 → 다음 단계로
            if event.key == pygame.K_SPACE and state["phase"] == "cleared":
                if stage_index < len(STAGES) - 1:   # 아직 단계가 남았으면
                    stage_index += 1                # 다음 단계로
                    state = reset_stage()           # 새 단계 준비
                    char_x = GAME_LEFT + 120        # 떡 위치 초기화
                    char_y = GROUND_Y
                    char_vy = 0
                    is_jumping = False
                else:                                # 마지막 단계였으면
                    state["phase"] = "allclear"      # 전체 클리어 화면으로
            # 게임오버에서 스페이스 → 현재 단계 재도전
            if event.key == pygame.K_SPACE and state["game_over"]:
                state = reset_stage()
                char_x = GAME_LEFT + 120
                char_y = GROUND_Y
                char_vy = 0
                is_jumping = False
            # r = 기준 자세 다시 잡기 (턱/볼 단계에서 유용)
            if event.key == pygame.K_r:
                base_pitch = None    # None으로 만들면 아래에서 다시 잡힘
                base_cheek = None

    # ─── (2) 웹캠 한 장 읽고 얼굴 인식 ───
    success, frame = cap.read()
    if not success:        # 웹캠 못 읽으면 건너뛰기
        continue
    frame = cv2.flip(frame, 1)                       # 거울처럼 좌우 뒤집기
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)     # 색 순서 맞추기
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)               # 얼굴 인식 실행

    # 인식 결과에서 필요한 것 꺼내기
    scores = {}        # 표정 점수
    lm = None          # 얼굴 점들
    pitch = None       # 머리 각도
    if result.face_blendshapes:
        scores = {b.category_name: b.score for b in result.face_blendshapes[0]}
    if result.face_landmarks:
        lm = result.face_landmarks[0]
    if result.facial_transformation_matrixes:
        # 머리 회전 정보에서 위아래 각도(pitch) 계산
        matrix = np.array(result.facial_transformation_matrixes[0])
        pitch = np.degrees(np.arctan2(-matrix[2][1], matrix[2][2]))

    # ─── (3) 안내 화면일 때 '평소 자세'를 기준으로 자동 저장 ───
    # (정면을 보고 있을 때의 각도/볼너비를 기준으로 잡아둠)
    if state["phase"] == "ready":
        if pitch is not None:
            base_pitch = pitch
        if lm is not None:
            face_height = dist(lm[10], lm[152])
            a = dist(lm[61], lm[135]) / face_height
            b = dist(lm[291], lm[364]) / face_height
            c = dist(lm[137], lm[366]) / face_height
            base_cheek = (a + b + c) / 3
            base_pucker_for_puff = scores.get("mouthPucker", 0)

    # ─── (4) 웹캠 영상을 게임 화면용으로 변환 ───
    frame_resized = cv2.resize(rgb, (HALF_W, WINDOW_H))
    frame_surface = pygame.image.frombuffer(frame_resized.tobytes(), (HALF_W, WINDOW_H), "RGB")

    # ─── (5) 현재 단계 동작이 감지되는지 확인 ───
    gesture_on = detect_gesture(current_stage["key"], scores, lm, pitch)

    # ====================================================
    #  화면: play (실제 달리기 게임 진행)
    # ====================================================
    if state["phase"] == "play":
        # 동작이 '꺼졌다가 막 켜진 순간'에만 점프 (계속 켜둬도 한 번만)
        if gesture_on and not gesture_was_on and not is_jumping and not state["game_over"]:
            char_vy = JUMP_POWER
            is_jumping = True

        # 계단 턱에 막혔는지 확인
        blocked = False
        char_right = char_x + CHAR_SIZE // 2
        char_bottom = char_y + CHAR_SIZE // 2
        for obs in state["obstacles"]:
            if obs["type"] == "stair":
                for i in range(3):
                    step_x = obs["x"] + i*STAIR_STEP_W
                    step_top = GROUND_Y - (i+1)*STAIR_STEP_H
                    if (step_x-6 <= char_right <= step_x+12 and char_bottom > step_top+8):
                        blocked = True

        # 막히지 않았으면 장애물 생성 & 이동
        if not state["game_over"] and not blocked:
            state["spawn_timer"] -= 1
            if state["spawn_timer"] <= 0:                       # 시간 되면 새 장애물
                obs_type = random.choice(["stick", "spike", "hole", "stair"])
                state["obstacles"].append({"x": WINDOW_W+20, "type": obs_type})
                state["spawn_timer"] = random.randint(OBSTACLE_GAP_MIN, OBSTACLE_GAP_MAX)
            new_obs = []
            for obs in state["obstacles"]:
                obs["x"] -= OBSTACLE_SPEED                      # 왼쪽으로 이동
                if obs["x"] > GAME_LEFT - 300:                  # 아직 화면 안이면 유지
                    new_obs.append(obs)
                else:                                           # 화면 밖으로 나가면
                    state["score"] += 1                         # 점수 +1
                    if state["score"] >= GOAL_SCORE:            # 목표 채우면
                        state["phase"] = "ending"               # 엔딩으로
            state["obstacles"] = new_obs

        # 중력 적용(떡 아래로 떨어뜨리기)
        char_vy += GRAVITY
        char_y += char_vy

        # 바닥/구멍/계단 판정
        current_ground = GROUND_Y
        over_hole = False
        for obs in state["obstacles"]:
            if obs["type"] == "hole":                           # 구멍 위인가
                if obs["x"] < char_x < obs["x"] + HOLE_W:
                    over_hole = True
            if obs["type"] == "stair":                          # 계단 위인가
                for i in range(3):
                    step_x = obs["x"] + i*STAIR_STEP_W
                    step_top = GROUND_Y - (i+1)*STAIR_STEP_H
                    if step_x-10 < char_x < step_x+STAIR_STEP_W+10:
                        if step_top < current_ground:
                            current_ground = step_top
        # 구멍에 빠지는 중인지
        falling_into_hole = over_hole and char_y > GROUND_Y and current_ground == GROUND_Y
        if falling_into_hole:
            if char_y >= WINDOW_H + 50:                          # 화면 밑으로 빠지면
                state["game_over"] = True                       # 게임오버
        else:
            # 바닥(또는 계단)에 닿으면 착지
            if char_y >= current_ground and not over_hole:
                char_y = current_ground
                char_vy = 0
                is_jumping = False
            elif char_y >= current_ground and over_hole and current_ground < GROUND_Y:
                char_y = current_ground
                char_vy = 0
                is_jumping = False

        # 가시/막대기에 부딪히면 게임오버
        char_rect = pygame.Rect(char_x-CHAR_SIZE//2, char_y-CHAR_SIZE//2, CHAR_SIZE, CHAR_SIZE)
        for obs in state["obstacles"]:
            if obs["type"] == "stick":
                if char_rect.colliderect(pygame.Rect(obs["x"], GROUND_Y-STICK_H//2+5, STICK_W, STICK_H)):
                    state["game_over"] = True
            elif obs["type"] == "spike":
                if char_rect.colliderect(pygame.Rect(obs["x"], GROUND_Y-SPIKE_H//2+15, SPIKE_W, SPIKE_H)):
                    state["game_over"] = True

        # 화면 그리기
        draw_game_world(state)
        draw_tteok(screen, char_x, int(char_y))
        # 상단에 단계/동작 안내
        info = font_small.render(f"Stage {stage_index+1}/6 - {current_stage['name']}", True, COLOR_GUIDE)
        screen.blit(info, (HALF_W+20, 15))
        sc = font_mid.render(f"{state['score']} / {GOAL_SCORE}", True, (40,40,60))
        screen.blit(sc, (HALF_W+20, 45))

        # 게임오버 표시
        if state["game_over"]:
            t = font_big.render("GAME OVER", True, COLOR_RED)
            screen.blit(t, (HALF_W+(HALF_W-t.get_width())//2, WINDOW_H//2-70))
            r = font_small.render("Press SPACE to retry", True, (40,40,60))
            screen.blit(r, (HALF_W+(HALF_W-r.get_width())//2, WINDOW_H//2+20))

    # ====================================================
    #  화면: ending (사람 등장 + 떡이 입으로 쏙)
    # ====================================================
    elif state["phase"] == "ending":
        # 남은 장애물 계속 흘려보내기
        new_obs = []
        for obs in state["obstacles"]:
            obs["x"] -= OBSTACLE_SPEED
            if obs["x"] > GAME_LEFT - 300:
                new_obs.append(obs)
        state["obstacles"] = new_obs

        # 사람이 오른쪽에서 슥 들어와 멈춤
        if state["person_x"] > PERSON_STOP_X:
            state["person_x"] -= OBSTACLE_SPEED
            if state["person_x"] < PERSON_STOP_X:
                state["person_x"] = PERSON_STOP_X
        person_stopped = state["person_x"] <= PERSON_STOP_X
        # 입 구멍 위치 계산
        mouth_x = state["person_x"] + MOUTH_OFFSET_X
        mouth_y = PERSON_Y + MOUTH_OFFSET_Y

        # 사람이 멈추면 떡이 입으로 빨려 들어감
        if person_stopped:
            char_x += (mouth_x - char_x) * 0.08   # 입쪽으로 조금씩 이동
            char_y += (mouth_y - char_y) * 0.08
            if abs(char_x-mouth_x) < 30 and abs(char_y-mouth_y) < 30:  # 입에 닿으면
                state["show_success"] = True
                state["phase"] = "cleared"        # 클리어 화면으로

        # 그리기
        draw_game_world(state)
        screen.blit(person_img, (int(state["person_x"]), PERSON_Y))
        if not state["show_success"]:
            draw_tteok(screen, int(char_x), int(char_y), 1.0)

    # ====================================================
    #  화면: cleared (단계 클리어 → 다음 단계로?)
    # ====================================================
    elif state["phase"] == "cleared":
        draw_game_world(state)
        screen.blit(person_img, (int(state["person_x"]), PERSON_Y))
        t = font_huge.render("Stage Clear!", True, COLOR_SUCCESS)
        screen.blit(t, (60, 80))
        if stage_index < len(STAGES) - 1:                    # 다음 단계가 있으면
            nxt = STAGES[stage_index+1]["name"]
            g = font_mid.render(f"Next: {nxt}", True, COLOR_GUIDE)
            screen.blit(g, (60, 200))
            r = font_small.render("Go to next stage? Press SPACE", True, COLOR_GUIDE)
            screen.blit(r, (60, 270))
        else:                                                # 마지막 단계였으면
            g = font_mid.render("Press SPACE to finish", True, COLOR_GUIDE)
            screen.blit(g, (60, 200))

    # ====================================================
    #  화면: allclear (6단계 전부 완료)
    # ====================================================
    elif state["phase"] == "allclear":
        screen.fill((255, 240, 245))
        t = font_huge.render("ALL CLEAR!", True, COLOR_SUCCESS)
        screen.blit(t, ((WINDOW_W-t.get_width())//2, 150))
        g = font_mid.render("All 6 exercises done!", True, COLOR_GUIDE)
        screen.blit(g, ((WINDOW_W-g.get_width())//2, 280))

    # ====================================================
    #  화면: ready (단계 시작 전 안내)
    # ====================================================
    if state["phase"] == "ready":
        draw_game_world(state)
        draw_tteok(screen, char_x, int(char_y))
        # 반투명 안내 박스
        box = pygame.Surface((HALF_W-40, 200))
        box.set_alpha(220)                # 220=약간 투명
        box.fill((255, 255, 255))
        screen.blit(box, (HALF_W+20, WINDOW_H//2 - 100))
        # 안내 글자들
        t1 = font_mid.render(f"Stage {stage_index+1}: {current_stage['name']}", True, COLOR_GUIDE)
        screen.blit(t1, (HALF_W+40, WINDOW_H//2 - 80))
        t2 = font_small.render("Look straight, then press SPACE", True, COLOR_GUIDE)
        screen.blit(t2, (HALF_W+40, WINDOW_H//2 - 20))
        t3 = font_small.render("(r: reset base pose)", True, (120,120,120))
        screen.blit(t3, (HALF_W+40, WINDOW_H//2 + 20))

    # ─── 이번 순간의 동작 상태를 저장(다음 순간 비교용) ───
    gesture_was_on = gesture_on

    pygame.display.flip()   # 그린 것을 실제 화면에 보이기
    clock.tick(30)          # 1초에 30번 반복(게임 속도)

# ─── 게임 종료 정리 ───
cap.release()           # 웹캠 끄기
detector.close()        # 얼굴 인식기 닫기
pygame.quit()           # pygame 종료