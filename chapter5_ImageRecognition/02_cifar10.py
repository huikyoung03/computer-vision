import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

# -----------------------------
# 1. CIFAR-10 데이터셋 로드
# -----------------------------
print("데이터셋 로드 중...")

# CIFAR-10 데이터셋 로드
# → 32x32 크기의 컬러 이미지 (RGB)
# → 10개의 클래스(비행기, 자동차, 개 등)
# → train / test 데이터 자동 분리
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# -----------------------------
# 2. 데이터 전처리 (정규화)
# -----------------------------
# 이미지 픽셀 값은 0~255 범위
# → 이를 255로 나누어 0~1 범위로 정규화
# → 학습 안정성 및 속도 향상
train_images, test_images = train_images / 255.0, test_images / 255.0

# CIFAR-10 클래스 이름 정의 (레이블 해석용)
class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# -----------------------------
# 3. CNN 모델 설계
# -----------------------------
model = models.Sequential([

   # Block 1
    layers.Input(shape=(32, 32, 3)),
    layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    # Block 2
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.30),

    # Block 3
    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.35),

    # Classifier
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# -----------------------------
# 모델 컴파일
# -----------------------------
model.compile(
    optimizer='adam',  # 최적화 알고리즘 (자동 학습률 조절)
    
    # sparse_categorical_crossentropy:
    # → 정답이 숫자 레이블일 때 사용하는 분류 손실 함수
    # → from_logits=True: 출력이 softmax 전 값(logits)임을 의미
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    
    metrics=['accuracy']  # 정확도 측정
)

# -----------------------------
# 4. 모델 훈련
# -----------------------------
print("모델 훈련 시작...")

# model.fit()
# → 학습 데이터로 모델을 훈련
# → epochs=10: 전체 데이터 10번 반복 학습
# → validation_data: 테스트 데이터를 이용해 성능 검증
model.fit(
    train_images, train_labels,
    epochs=10,
    validation_data=(test_images, test_labels)
)

# -----------------------------
# 5. 모델 성능 평가
# -----------------------------
# test 데이터로 최종 정확도 측정
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print(f'\n테스트 정확도: {test_acc:.4f}')

# -----------------------------
# 6. dog.jpg 이미지 예측 함수
# -----------------------------
def predict_local_image(img_path):
    try:
        # 이미지 로드 및 크기 조정
        # → CIFAR-10 모델 입력 크기 (32x32)로 resize
        img = image.load_img(img_path, target_size=(32, 32))

        # 이미지 → numpy 배열 변환
        img_array = image.img_to_array(img)

        # 배치 차원 추가 (모델 입력 형태 맞춤)
        # (32,32,3) → (1,32,32,3)
        img_array = np.expand_dims(img_array, axis=0)

        # 정규화 (0~255 → 0~1)
        img_array /= 255.0

        # 모델 예측 수행
        predictions = model.predict(img_array)

        # logits → 확률 변환 (softmax)
        score = tf.nn.softmax(predictions[0])
        
        # 가장 높은 확률을 가진 클래스 선택
        predicted_class = class_names[np.argmax(score)]

        # 예측 결과 출력
        print(f"\n--- 예측 결과 ---")
        print(f"이미지 경로: {img_path}")
        print(f"예측 결과: {predicted_class} ")
        
    except Exception as e:
        print(f"이미지를 불러오는 중 오류 발생: {e}")

# -----------------------------
# 7. dog.jpg 예측 실행
# -----------------------------
# 같은 폴더에 있는 dog.jpg 파일을 불러와 예측 수행
predict_local_image('dog.jpg')