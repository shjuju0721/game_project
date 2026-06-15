import cv2              # OpenCV: 카메라 영상 처리 및 화면 출력용
import mediapipe as mp  # MediaPipe: 얼굴/포즈 인식 라이브러리

# MediaPipe의 주요 모듈을 미리 변수로 지정
mp_drawing = mp.solutions.drawing_utils          # 랜드마크(점)를 화면에 그려주는 도구
mp_drawing_styles = mp.solutions.drawing_styles  # 그리기 기본 스타일(색/굵기)
mp_face_mesh = mp.solutions.face_mesh             # 얼굴 메쉬(468개 점) 인식 모델

cap = cv2.VideoCapture(0)  # 0번 카메라(기본 웹캠)를 연다

# with 문으로 FaceMesh 모델 설정 (블록이 끝나면 자동으로 자원 해제)
with mp_face_mesh.FaceMesh(
    static_image_mode = False,        # False = 동영상 모드(연속 프레임 추적)
    max_num_faces = 1,                # 감지할 최대 얼굴 수
    refine_landmarks = True,          # True = 눈동자/입술 주변을 더 정밀하게 추적
    min_detection_confidence = 0.5,   # 얼굴 감지 최소 신뢰도
    min_tracking_confidence = 0.5     # 추적 유지 최소 신뢰도
) as face_mesh:
    
    while cap.isOpened():  # 카메라가 열려 있는 동안 반복
        success, frame = cap.read()  # 한 프레임 읽기. success=성공 여부, frame=영상
        if not success:
            print("카메라를 찾을 수 없습니다.")
            continue  # 프레임을 못 읽으면 다음 반복으로

        # 성능 향상을 위해 처리 전 이미지를 읽기 전용으로 설정
        frame.flags.writeable = False
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # OpenCV(BGR) → MediaPipe(RGB) 변환
        results = face_mesh.process(image)              # 얼굴 인식 실행 → 랜드마크 반환

        # 화면에 그리기 위해 다시 쓰기 가능 + 색상 원래대로(BGR) 복구
        frame.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 얼굴이 감지되었을 때만 실행 (여러 얼굴 가능하므로 반복)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # 1) 얼굴 메쉬(삼각형 그물망) 그리기
                mp_drawing.draw_landmarks(
                    image = image,
                    landmark_list = face_landmarks,
                    connections = mp_face_mesh.FACEMESH_TESSELATION,  # 그물망 연결선
                    landmark_drawing_spec = None,                     # 점은 안 그림(선만)
                    connection_drawing_spec = mp_drawing_styles
                        .get_default_face_mesh_tesselation_style()
                )
                # 2) 얼굴 윤곽선(눈/눈썹/입술/얼굴형) 그리기
                mp_drawing.draw_landmarks(
                    image = image,
                    landmark_list = face_landmarks,
                    connections = mp_face_mesh.FACEMESH_CONTOURS,     # 윤곽 연결선
                    landmark_drawing_spec = None,
                    connection_drawing_spec = mp_drawing_styles
                        .get_default_face_mesh_contours_style()
                )
                # 3) 눈동자(홍채) 그리기 — refine_landmarks=True일 때만 동작
                mp_drawing.draw_landmarks(
                    image = image,
                    landmark_list = face_landmarks,
                    connections = mp_face_mesh.FACEMESH_IRISES,       # 홍채 연결선
                    landmark_drawing_spec = None,
                    connection_drawing_spec = mp_drawing_styles
                        .get_default_face_mesh_iris_connections_style()
                )

        # 거울처럼 좌우 반전하면 사용하기 더 자연스러움 (선택 사항)
        cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))

        # 5ms 대기하며 키 입력 확인. 'q'를 누르면 반복 종료
        if cv2.waitKey(5) & 0xFF == ord("q"):
            break


cap.release()              # 카메라 자원 해제
cv2.destroyAllWindows()    # 열린 모든 창 닫기