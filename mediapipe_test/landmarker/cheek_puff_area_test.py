"""
볼 부풀리기 검증용 - 면적 + z좌표(깊이) 결합 방식
==================================================
1번(면적, 범위 확장) + 2번(z좌표 깊이)을 같이 측정.
화면에 각각의 증가율과 둘을 합친 종합 점수를 띄워서
어떤 신호가 제일 잘 갈리는지 비교한다.

조작:
  q : 종료
  r : baseline(평상시) 값 기록  ← 정색한 평소 표정일 때 한 번만!

필요한 파일:
  face_landmarker.task  (같은 폴더 또는 MODEL_PATH 수정)
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = r"face_landmarker.task"


# ── 볼 랜드마크 (1번: 범위 더 넓게 - 입꼬리~귀밑 라인까지 포함) ──
LEFT_CHEEK  = [205, 36, 142, 100, 50, 101, 118, 117, 123, 147, 213, 192, 214]
RIGHT_CHEEK = [425, 266, 371, 329, 280, 330, 347, 346, 352, 376, 433, 416, 434]

# z좌표(깊이) 측정용 - 볼 중심부 점들
LEFT_CHEEK_Z  = [50, 101, 118, 117, 205]
RIGHT_CHEEK_Z = [280, 330, 347, 346, 425]

# 얼굴 폭 정규화용 (양 광대뼈 바깥)
FACE_LEFT  = 234
FACE_RIGHT = 454

# z 평활화 계수 (0~1, 작을수록 부드럽지만 느림)
SMOOTH = 0.3


def polygon_area(points):
    x, y = points[:, 0], points[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def main():
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
    landmarker = vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    base = {}                # baseline 저장
    sm_zl = sm_zr = None      # z 평활화 상태

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        key = cv2.waitKey(1) & 0xFF

        if result.face_landmarks:
            lm = result.face_landmarks[0]
            pts = np.array([[p.x * w, p.y * h] for p in lm])
            zs  = np.array([p.z for p in lm])

            face_w = max(np.linalg.norm(pts[FACE_LEFT] - pts[FACE_RIGHT]), 1e-6)

            # ── 1번: 면적 (얼굴폭 제곱으로 정규화) ──
            left_area  = polygon_area(pts[LEFT_CHEEK])  / (face_w ** 2)
            right_area = polygon_area(pts[RIGHT_CHEEK]) / (face_w ** 2)

            # ── 2번: z좌표 (볼 평균 z, 얼굴폭으로 정규화 → 거리 영향 줄임) ──
            # MediaPipe z는 카메라 쪽이 음수. 부풀리면 더 작아짐(앞으로 나옴).
            # 코(점1) 기준 상대 z를 써서 머리 전체 이동 영향 제거.
            nose_z = zs[1]
            zl_raw = (np.mean(zs[LEFT_CHEEK_Z])  - nose_z) / face_w
            zr_raw = (np.mean(zs[RIGHT_CHEEK_Z]) - nose_z) / face_w

            # 평활화
            if sm_zl is None:
                sm_zl, sm_zr = zl_raw, zr_raw
            else:
                sm_zl = SMOOTH * zl_raw + (1 - SMOOTH) * sm_zl
                sm_zr = SMOOTH * zr_raw + (1 - SMOOTH) * sm_zr

            # 볼 다각형 시각화
            for idxs, color in [(LEFT_CHEEK, (0, 255, 0)), (RIGHT_CHEEK, (0, 200, 255))]:
                cv2.polylines(frame, [pts[idxs].astype(np.int32)], True, color, 1)

            cv2.putText(frame, f"areaL {left_area:.4f}  areaR {right_area:.4f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)

            if base:
                # 면적 증가율(%)
                dAL = (left_area  / base['aL'] - 1) * 100
                dAR = (right_area / base['aR'] - 1) * 100
                # z 변화량 (부풀리면 z가 작아지므로 baseline - 현재 = 양수일수록 부풀림)
                dZL = (base['zL'] - sm_zl) / abs(base['zL'] + 1e-6) * 100
                dZR = (base['zR'] - sm_zr) / abs(base['zR'] + 1e-6) * 100
                # 종합 점수 (면적 + z 단순 합, 좌우 평균)
                score = ((dAL + dAR) / 2) + ((dZL + dZR) / 2)

                cv2.putText(frame, f"dArea  L{dAL:+.1f}%  R{dAR:+.1f}%", (10, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(frame, f"dZ     L{dZL:+.1f}%  R{dZR:+.1f}%", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"SCORE  {score:+.1f}", (10, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 100, 255), 2)
            else:
                cv2.putText(frame, "press 'r' (neutral face)", (10, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if key == ord('r'):
                base = {'aL': left_area, 'aR': right_area,
                        'zL': sm_zl, 'zR': sm_zr}
                print(f"[baseline] areaL={left_area:.4f} areaR={right_area:.4f} "
                      f"zL={sm_zl:.4f} zR={sm_zr:.4f}")
        else:
            cv2.putText(frame, "no face", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        if key == ord('q'):
            break

        cv2.imshow("Cheek Puff Test  area+z  (r=baseline, q=quit)", frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()