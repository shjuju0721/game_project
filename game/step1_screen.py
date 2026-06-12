# ============================================================
#  STEP 1 : 게임 화면의 '토대' 만들기
#  목표 - 창을 좌우로 나눠서, 왼쪽엔 내 얼굴(웹캠), 오른쪽엔 빈 게임판을 띄운다
# ============================================================

# ─── 필요한 도구(라이브러리) 불러오기 ───
import pygame              # 게임 화면을 만들고 그림을 그리는 도구
import cv2                 # 웹캠을 켜고 영상을 다루는 도구 (OpenCV)
import mediapipe as mp     # 얼굴을 인식하는 도구 (구글)


# ============================================================
#  1. 기본 설정값 정하기 (숫자를 맨 위에 모아두면 나중에 고치기 쉬움)
# ============================================================

WINDOW_W, WINDOW_H = 1280, 480   # 창 크기: 가로 1280, 세로 480 (가로로 길쭉하게)
HALF_W = WINDOW_W // 2           # 가로의 절반(640). 왼쪽/오른쪽 경계.
                                 # '//'는 나눈 뒤 소수점 버리고 정수만 (640.0이 아니라 640)


# ============================================================
#  2. pygame(게임 화면) 준비하기
# ============================================================

pygame.init()                                          # pygame 시동 (항상 맨 처음)
screen = pygame.display.set_mode((WINDOW_W, WINDOW_H)) # 창 만들기. screen = 그림 그릴 도화지
pygame.display.set_caption("꿀떡 꿀떡 게임")            # 창 제목
clock = pygame.time.Clock()                            # 게임 속도(프레임) 조절용 시계


# ============================================================
#  3. 얼굴 인식 + 웹캠 준비하기
# ============================================================

mp_face_mesh = mp.solutions.face_mesh              # 얼굴 점 468개 찾는 기능
mp_drawing = mp.solutions.drawing_utils            # 점/선을 그려주는 기능
mp_drawing_styles = mp.solutions.drawing_styles    # 그릴 때의 색·굵기 기본 스타일

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,              # 얼굴 1명만
    refine_landmarks=True,        # 입술·눈 정밀하게 (재활엔 입이 중요)
    min_detection_confidence=0.5, # 얼굴이라 확신하는 최소 기준
    min_tracking_confidence=0.5   # 찾은 얼굴 계속 따라가는 최소 기준
)

cap = cv2.VideoCapture(0)         # 0번 카메라(기본 웹캠) 켜기


# ============================================================
#  4. 색깔 미리 정해두기 (색 = 빨강R 초록G 파랑B 를 0~255로 섞음)
# ============================================================

COLOR_GAME_BG = (30, 30, 50)      # 오른쪽 게임 배경색 (어두운 남색)


# ============================================================
#  5. 게임의 심장 = 반복문(while)
#  게임은 '매 순간 화면을 다시 그리는' 일을 1초에 수십 번 반복함
# ============================================================

running = True                    # 게임이 돌아가는 중인지 표시하는 스위치

while running:                    # running이 True인 동안 무한 반복

    # ─── (1) 사용자의 행동(이벤트) 확인 ───
    for event in pygame.event.get():           # 그동안 일어난 사건 하나씩 확인
        if event.type == pygame.QUIT:          # 창의 X(닫기) 버튼을 눌렀으면
            running = False                    # 스위치 꺼서 종료 준비
        if event.type == pygame.KEYDOWN and event.key == pygame.K_q:  # 'q' 키 눌렀으면
            running = False                    # 마찬가지로 종료

    # ─── (2) 웹캠에서 사진 한 장 가져오기 ───
    success, frame = cap.read()    # success=잘 읽었는지, frame=실제 사진
    if not success:
        print("웹캠을 못 읽음! 카메라가 다른 곳에서 사용 중일 수 있어요.")
        continue                   # 못 읽으면 이번 반복 건너뛰기

    frame = cv2.flip(frame, 1)     # 좌우 반전 (거울처럼 보이게)

    # ─── (3) 얼굴 인식 돌리기 ───
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 색 순서 변환 (OpenCV=BGR → MediaPipe=RGB)
    results = face_mesh.process(rgb)              # 얼굴 인식 실행

    # ─── (4) 인식된 얼굴에 윤곽선 그리기 ───
    if results.multi_face_landmarks:              # 얼굴이 찾아졌으면
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,                      # frame(사진) 위에 그림
                landmark_list=face_landmarks,     # 찾은 얼굴 점들
                connections=mp_face_mesh.FACEMESH_CONTOURS,  # 눈·입·얼굴형 윤곽선
                landmark_drawing_spec=None,       # 점은 안 그림 (선만)
                connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_contours_style()
            )

    # ─── (5) 웹캠 사진을 pygame 그림으로 변환 ───
    # ⚠️ OpenCV 사진과 pygame 그림은 데이터 정리 방식이 달라서 맞춰줘야 함
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)        # 색 RGB로
    frame_resized = cv2.resize(frame_rgb, (HALF_W, WINDOW_H)) # 왼쪽 절반 크기(640x480)로
    frame_surface = pygame.image.frombuffer(                 # pygame 그림으로 변환
        frame_resized.tobytes(),    # 사진 데이터를 바이트로
        (HALF_W, WINDOW_H),         # 크기 (가로, 세로)
        "RGB"                       # 색 형식
    )

    # ─── (6) 실제로 화면에 그리기 (뒤에서 앞으로 덮어 그림) ───
    screen.fill(COLOR_GAME_BG)             # 1. 도화지 전체를 배경색으로 칠함
    screen.blit(frame_surface, (0, 0))     # 2. 왼쪽 위(0,0)에 웹캠 영상 붙임

    # 가운데 경계선 (왼쪽 게임 / 오른쪽 게임 구분 흰 세로선)
    pygame.draw.line(
        screen, (255, 255, 255),    # 흰색
        (HALF_W, 0), (HALF_W, WINDOW_H),  # 가운데 위 → 가운데 아래
        2                           # 굵기 2픽셀
    )

    pygame.display.flip()      # 지금까지 그린 걸 실제 화면에 반영 (이게 있어야 보임)
    clock.tick(30)             # 1초에 30번만 반복 (속도 조절)


# ============================================================
#  6. 게임 종료 뒷정리 (while을 빠져나오면 실행)
# ============================================================

cap.release()        # 웹캠 끄기
face_mesh.close()    # 얼굴 인식기 닫기
pygame.quit()        # pygame 창 닫기