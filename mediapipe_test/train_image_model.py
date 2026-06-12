import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
import matplotlib.pyplot as plt

# ─── 설정 ───
DATA_DIR = "dataset"
IMG_SIZE = (160, 160)
BATCH_SIZE = 16
EPOCHS_STAGE1 = 25      # 1단계: 얼린 채로 기본 학습
EPOCHS_STAGE2 = 15      # 2단계: 미세조정 추가 학습

# ─── 1) 데이터 불러오기 ───
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="training",
    seed=123, image_size=IMG_SIZE, batch_size=BATCH_SIZE
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="validation",
    seed=123, image_size=IMG_SIZE, batch_size=BATCH_SIZE
)
class_names = train_ds.class_names
print("분류할 동작:", class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# ─── 2) 데이터 증강 ───
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomBrightness(0.1),
])

# ─── 3) MobileNetV2 가져오기 ───
base_model = MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False   # 1단계에서는 얼려둠

# ─── 4) 모델 조립 ───
model = models.Sequential([
    data_augmentation,
    layers.Rescaling(1./127.5, offset=-1),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ─── 5) 1단계 학습 (얼린 채로) ───
print("\n=== 1단계 학습 시작 ===")
history1 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_STAGE1)

# ─── 6) 2단계: 미세조정 ───
# MobileNet의 뒷부분만 풀어줌 (앞부분은 일반적인 특징이라 그대로 두는 게 좋음)
print("\n=== 2단계 미세조정 시작 ===")
base_model.trainable = True
# 전체 154개 층 중 뒤쪽 일부만 학습. 앞 100층은 그대로 얼려둠.
for layer in base_model.layers[:100]:
    layer.trainable = False

# 미세조정은 아주 작은 학습률로! (기존 지식을 망가뜨리지 않도록)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # 1단계보다 100배 작게
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# 1단계에 이어서 계속 학습 (initial_epoch로 이어붙임)
history2 = model.fit(
    train_ds, validation_data=val_ds,
    epochs=EPOCHS_STAGE1 + EPOCHS_STAGE2,
    initial_epoch=EPOCHS_STAGE1
)

# ─── 7) 그래프 (1단계 + 2단계 이어서) ───
acc = history1.history["accuracy"] + history2.history["accuracy"]
val_acc = history1.history["val_accuracy"] + history2.history["val_accuracy"]
plt.plot(acc, label="train accuracy")
plt.plot(val_acc, label="validation accuracy")
plt.axvline(x=EPOCHS_STAGE1 - 1, color="gray", linestyle="--", label="fine-tuning starts")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training Progress (with fine-tuning)")
plt.savefig("training_result.png")
print("\n그래프를 'training_result.png'로 저장했습니다.")

# ─── 8) 모델 저장 ───
model.save("puff_model.keras")
print("모델을 'puff_model.keras'로 저장했습니다.")
print("분류 순서:", class_names)