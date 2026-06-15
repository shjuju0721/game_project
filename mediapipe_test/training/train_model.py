import pandas as pd                                      # csv 읽기
from sklearn.model_selection import train_test_split     # 데이터를 학습용/시험용으로 나누기
from sklearn.ensemble import RandomForestClassifier      # 분류 모델 (입문용으로 좋고 강력함)
from sklearn.metrics import classification_report, confusion_matrix  # 성적표
import joblib                                             # 학습된 모델을 파일로 저장

CSV_FILE = "face_data.csv"
MODEL_FILE = "face_model.pkl"   # 학습 완료된 모델이 저장될 파일

# ─── 1) 데이터 불러오기 (헤더/열 개수 문제에 강한 안전 버전) ───
# refine_landmarks=True면 점이 478개라 헤더(468개)와 열 수가 안 맞음.
# 그래서 헤더를 무시하고 첫 열만 라벨로, 나머지 전부를 좌표로 읽음.
df = pd.read_csv(CSV_FILE, header=None, skiprows=1)   # 헤더 줄 건너뛰고 통째로 읽기
df = df.rename(columns={0: "label"})                  # 첫 열 = 라벨(정답)

# 혹시 수집을 여러 번 해서 중간에 헤더 줄이 또 끼어든 경우 제거
df = df[df["label"].isin(["neutral", "open", "puff", "suck"])]

# 좌표 열들을 숫자로 변환 (글자가 섞여 있었으면 깨끗하게 정리됨)
coord_cols = df.columns[1:]
df[coord_cols] = df[coord_cols].apply(pd.to_numeric, errors="coerce")
df = df.dropna()  # 변환 안 되는 이상한 줄 제거

print("전체 데이터 개수:", len(df))
print("동작별 개수:\n", df["label"].value_counts())

# ─── 2) 입력(X)과 정답(y) 나누기 ───
X = df.drop("label", axis=1)   # 좌표 전부 = 입력
y = df["label"]                # 동작 이름 = 정답

# ─── 3) 학습용 / 시험용 분리 ───
# 모델이 못 본 데이터로 평가해야 진짜 실력을 알 수 있음 → 20%는 시험용으로 떼어둠
# stratify=y : 네 동작이 학습/시험 양쪽에 골고루 들어가게 함
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n학습용 {len(X_train)}개, 시험용 {len(X_test)}개")

# ─── 4) 모델 학습 ───
# RandomForest = 여러 개의 '결정 트리'가 투표해서 답을 정하는 방식. 안정적이고 빠름.
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)   # 이 한 줄이 '학습' 그 자체
print("\n학습 완료!")

# ─── 5) 평가: 시험용 데이터로 얼마나 맞히나 ───
y_pred = model.predict(X_test)

print("\n=== 동작별 성적표 ===")
print(classification_report(y_test, y_pred))

print("=== 혼동 행렬 (어떤 동작을 무엇으로 헷갈렸나) ===")
labels = sorted(y.unique())
print("순서:", labels)
print(confusion_matrix(y_test, y_pred, labels=labels))

# ─── 6) 모델 저장 ───
joblib.dump(model, MODEL_FILE)
print(f"\n모델을 '{MODEL_FILE}'로 저장했습니다.")