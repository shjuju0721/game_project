# 🍡 꿀떡 꿀떡 게임 (Face Rehabilitation Game)

웹캠으로 **얼굴 표정 / 입 동작을 인식**해서 즐기는 재활(rehabilitation)용 게임 프로젝트입니다.
입 벌리기·볼 부풀리기 같은 구강 안면 운동을 게임처럼 유도하는 것이 목표이며,
이 저장소에는 **완성된 게임 화면(`game/`)** 과 거기에 도달하기까지의 **실험·모델 학습 코드(`mediapipe_test/`)** 가 함께 들어 있습니다.

> 핵심 기술: `MediaPipe` (얼굴 랜드마크 468점 / 블렌드셰이프) · `OpenCV` (웹캠·영상처리) · `pygame` (게임 화면) · `scikit-learn` / `TensorFlow` (표정 분류 모델)

---

## 📂 폴더 구조

```
embeded/
├── game/                  # 🎮 게임 본체
│   └── step1_screen.py        # STEP 1: 게임 화면 토대 (좌: 웹캠+얼굴인식 / 우: 게임판)
│
└── mediapipe_test/        # 🧪 실험 · 모델 학습 코드
    ├── 데이터 수집
    ├── 모델 학습 / 실행
    ├── 표정 검증 실험
    └── 기초 학습용 예제
```

> 참고: 학습 데이터셋 원본(`dataset/`)은 용량 문제로 저장소에 포함하지 않았습니다.

---

## 🎮 game — 게임 본체

| 파일 | 설명 |
|------|------|
| `step1_screen.py` | 게임의 **1단계 토대**. 창을 좌우로 나눠 **왼쪽엔 웹캠 + 실시간 얼굴 윤곽선**, 오른쪽엔 빈 게임판을 띄웁니다. `pygame` 게임 루프와 `MediaPipe FaceMesh`를 연결하는 기본 골격입니다. |

**조작법**
- `Q` 키 또는 창 닫기(X) → 종료

**실행**
```bash
cd game
python step1_screen.py
```

---

## 🧪 mediapipe_test — 실험 & 모델 학습

게임에 쓸 "얼굴 동작 인식" 방법을 찾기 위한 실험들입니다. 크게 **두 가지 인식 방식**을 시도했습니다.

### ① 랜드마크 좌표 기반 (scikit-learn / RandomForest)

얼굴 468개 점의 좌표를 숫자로 바꿔 **머신러닝 모델**로 동작을 분류합니다.
동작: `neutral`(무표정) · `open`(입 벌림) · `puff`(볼 부풀리기) · `suck`(볼 홀쭉)

| 파일 | 역할 |
|------|------|
| `mediapipe_face_data.py` | **데이터 수집** — 숫자키(0~3)를 누르는 동안 좌표를 `face_data.csv`에 기록. 코끝 기준 상대좌표 + 얼굴폭 정규화로 위치·크기 무관하게 만듦 |
| `train_model.py` | **모델 학습** — `face_data.csv`로 RandomForest 학습 → `face_model.pkl` 저장, 성적표·혼동행렬 출력 |
| `run_model.py` | **실시간 인식** — `face_model.pkl`을 불러와 웹캠에서 동작을 실시간 예측·표시 |

```bash
python mediapipe_face_data.py   # 1) 데이터 모으기
python train_model.py           # 2) 학습 (→ face_model.pkl)
python run_model.py             # 3) 실시간 인식
```

### ② 이미지 기반 딥러닝 (TensorFlow / MobileNetV2)

얼굴 **사진 자체**를 CNN(전이학습)으로 분류합니다. (`neutral` vs `puff` 2분류)

| 파일 | 역할 |
|------|------|
| `collect_images.py` | **사진 수집** — `dataset/<동작>/` 폴더에 얼굴 사진 저장 (스페이스바/`a`) |
| `train_image_model.py` | **모델 학습** — MobileNetV2 전이학습 + 미세조정(fine-tuning) → `puff_model.keras` 저장, 학습 그래프 `training_result.png` |
| `test_puff.py` | **실시간 판정** — `puff_model.keras`로 볼 부풀리기(PUFF/NEUTRAL) 실시간 확률 표시 |

```bash
python collect_images.py        # 1) 사진 모으기
python train_image_model.py     # 2) 학습 (→ puff_model.keras)
python test_puff.py             # 3) 실시간 판정
```

### ③ 표정 검증 실험

| 파일 | 역할 |
|------|------|
| `landmarker_test.py` | MediaPipe **Face Landmarker**의 블렌드셰이프(표정 수치 52종) 체험 — `jawOpen`, `cheekPuff` 등을 막대그래프로 확인 |
| `cheek_puff_area_test.py` | **볼 부풀리기 정밀 검증** — 볼 다각형 **면적 + z좌표(깊이)** 를 결합해 부풀림 정도를 점수화. `r`로 평상시(baseline) 기록 후 증가율 비교 |

### ④ 기초 학습용 예제

OpenCV / MediaPipe 입문용 작은 스크립트들입니다.

| 파일 | 역할 |
|------|------|
| `webcam_read.py` | 웹캠 영상 띄우기 (가장 기본) |
| `webcam_face.py` | Haar Cascade로 얼굴 박스 검출 |
| `mediapipe_face.py` | FaceMesh로 얼굴 그물망·윤곽·홍채 그리기 |
| `mediapipe_test.py` | MediaPipe **Pose**로 팔 각도 계산 (관절 각도 측정 예제) |
| `image_resize.py` | 이미지 리사이즈·자르기 |
| `image_blurred.py` | 가우시안 블러 적용 |

---

## 📦 포함된 모델 · 데이터 파일

| 파일 | 설명 |
|------|------|
| `face_landmarker.task` | MediaPipe Face Landmarker 모델 (블렌드셰이프용) |
| `haarcascade_frontalface_default.xml` | OpenCV 얼굴 검출용 Haar Cascade |
| `face_data.csv` | 수집된 랜드마크 좌표 학습 데이터 |
| `face_model.pkl` | 학습된 RandomForest 모델 |
| `puff_model.keras` | 학습된 MobileNetV2 모델 |
| `training_result.png` | 딥러닝 학습 정확도 그래프 |
| `woo.jpg` | 이미지 처리 예제용 샘플 사진 |

---

## ⚙️ 설치 & 실행 환경

- **Python 3.x** + **웹캠**

```bash
pip install opencv-python mediapipe pygame numpy pandas scikit-learn joblib tensorflow matplotlib
```

| 라이브러리 | 용도 |
|-----------|------|
| `opencv-python` | 웹캠·영상처리 |
| `mediapipe` | 얼굴 랜드마크·블렌드셰이프 인식 |
| `pygame` | 게임 화면 |
| `numpy` / `pandas` | 수치·데이터 처리 |
| `scikit-learn` / `joblib` | 랜드마크 기반 분류 모델 (①) |
| `tensorflow` / `matplotlib` | 이미지 기반 딥러닝 모델 (②) |

> 💡 대부분의 스크립트는 0번(기본) 웹캠을 사용합니다. 실행 중 종료는 영상 창에서 `q` 키입니다.

---

## 🗺️ 개발 흐름 한눈에 보기

```
기초 예제(webcam/face/pose)  →  FaceMesh·Landmarker로 얼굴 인식 검증
        →  "어떤 동작을 어떻게 잡을까?" 두 갈래 실험
              ├─ ① 랜드마크 좌표 + RandomForest
              └─ ② 얼굴 사진 + MobileNetV2 딥러닝
        →  볼 부풀리기 정밀 검증(면적+깊이)
        →  🎮 game/step1_screen.py 로 게임 화면 토대 구축
```
