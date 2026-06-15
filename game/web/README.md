# 꿀떡 꿀떡 게임 (웹 버전)

`step5b_ending.py`의 흐름·규칙·엔딩을 브라우저에서 실행할 수 있게 옮긴 버전입니다.

## 실행 방법

브라우저에서 카메라 사용과 모듈 로딩을 하려면 `http://localhost`로 열어야 합니다
(파일을 더블클릭해 `file://`로 열면 카메라/모듈이 차단됩니다).

1. 이 폴더(`game/web`)에서 로컬 서버 실행:
   ```powershell
   python -m http.server 8000
   ```
2. 브라우저에서 접속: <http://localhost:8000/index.html>

## 조작

- **카메라 켜고 시작**: 웹캠을 보며 **입을 벌리면** 떡이 점프합니다 (원작과 동일, MediaPipe FaceLandmarker의 `jawOpen` 사용).
- **카메라 없이 시작(키보드)**: **스페이스바** 또는 **화면 클릭**으로 점프.
- 게임오버 / 성공 화면에서 **스페이스바**로 재시작.

## 게임 규칙 (원작과 동일)

- 화면 좌측 = 웹캠, 우측 = 게임 (1280×480).
- 장애물 4종: 막대(stick) · 가시(spike) · 구멍(hole) · 계단(stair).
- 막대/가시에 닿거나 구멍에 빠지면 GAME OVER.
- **5점**을 모으면 엔딩: 오른쪽에서 사람이 들어오고 떡이 입으로 빨려 들어가 **Success! / Stage 1 Clear**.

## 파일

- `index.html` — 게임 전체 (HTML/CSS/JS, 단일 파일)
- `ending_image.png` — 엔딩에 등장하는 입 벌린 사람 이미지
- MediaPipe 얼굴 인식 모델/런타임은 CDN(jsdelivr, Google)에서 로드합니다 (인터넷 필요).
