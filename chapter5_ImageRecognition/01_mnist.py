import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# -----------------------------
# 1. MNIST 데이터셋 로드
# -----------------------------
# datasets.mnist.load_data()
# → MNIST 손글씨 숫자 데이터셋을 불러옴
# → 학습용 데이터와 테스트용 데이터를 자동으로 분리해서 반환
# → x_train, x_test : 이미지 데이터
# → y_train, y_test : 각 이미지에 해당하는 정답 레이블(0~9)
# → 이미지 크기는 28x28, 흑백(1채널) 이미지
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

# -----------------------------
# 2. 데이터 전처리
# -----------------------------
# MNIST 이미지의 픽셀 값은 0~255 범위의 정수값
# 이를 255.0으로 나누어 0~1 범위로 정규화
# → 학습 속도 향상
# → 수치 안정성 증가
# → 신경망이 더 효율적으로 학습 가능
x_train = x_train / 255.0
x_test = x_test / 255.0

# -----------------------------
# 3. 신경망 모델 구축
# -----------------------------
# Sequential 모델:
# → 층을 순서대로 쌓는 가장 기본적인 신경망 구조
model = models.Sequential([
    
    # Flatten
    # → 28x28 형태의 2차원 이미지를 784(=28*28) 크기의 1차원 벡터로 변환
    # → Dense(완전연결층)에 입력하기 위해 차원을 펼쳐주는 역할
    layers.Flatten(input_shape=(28, 28)),

    # Dense(128, activation='relu')
    # → 은닉층(hidden layer)
    # → 128개의 뉴런 사용
    # → ReLU 활성화 함수:
    #    f(x) = max(0, x)
    # → 비선형성을 추가하여 복잡한 패턴 학습 가능
    layers.Dense(128, activation='relu'),

    # Dense(10, activation='softmax')
    # → 출력층(output layer)
    # → 숫자 0~9 총 10개 클래스를 분류하므로 노드 수는 10
    # → softmax:
    #    각 클래스에 대한 확률값으로 변환
    #    출력값들의 합은 1이 됨
    layers.Dense(10, activation='softmax')
])

# -----------------------------
# 4. 모델 컴파일
# -----------------------------
# optimizer='adam'
# → 대표적인 최적화 알고리즘
# → 학습률을 자동 조정하며 빠르고 안정적으로 학습 가능
#
# loss='sparse_categorical_crossentropy'
# → 다중 클래스 분류 문제에서 사용하는 손실 함수
# → 정답 레이블이 원-핫 인코딩이 아닌 정수 형태일 때 사용
#
# metrics=['accuracy']
# → 학습 중 정확도를 함께 출력
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# -----------------------------
# 5. 모델 학습
# -----------------------------
# model.fit(...)
# → 학습 데이터를 이용하여 모델을 훈련
#
# epochs=7
# → 전체 학습 데이터를 7번 반복 학습
#
# batch_size=64
# → 한 번에 64장씩 데이터를 나누어 학습
# → 너무 크면 메모리 부담, 너무 작으면 학습 불안정 가능
#
# history
# → 학습 과정에서의 loss, accuracy 값이 저장됨
# → 이후 그래프 시각화에 사용
history = model.fit(x_train, y_train, epochs=7, batch_size=64)

# -----------------------------
# 6. 모델 평가
# -----------------------------
# model.evaluate(...)
# → 테스트 데이터로 최종 성능 평가
#
# verbose=0
# → 평가 진행 로그를 간단히 숨김
#
# test_loss : 테스트 데이터에서의 손실값
# test_acc  : 테스트 데이터에서의 정확도
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

# 소수점 4자리까지 정확도 출력
print(f"정확도: {test_acc:.4f}")

# -----------------------------
# 7. 정확도 / 손실 시각화
# -----------------------------
# history.history['accuracy']
# → 각 epoch마다의 학습 정확도 저장
#
# history.history['loss']
# → 각 epoch마다의 학습 손실값 저장
#
# epoch 번호를 1부터 시작하도록 range 생성
epochs = range(1, len(history.history['accuracy']) + 1)

# 전체 figure 크기 설정
plt.figure(figsize=(12, 5))

# -----------------------------
# 7-1. 정확도 그래프
# -----------------------------
plt.subplot(1, 2, 1)
# → 1행 2열 중 첫 번째 위치에 그래프 배치

plt.plot(epochs, history.history['accuracy'], marker='o')
# → epoch별 accuracy 값을 선 그래프로 그림
# → marker='o' : 각 epoch 위치를 점으로 표시

plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)
# → 격자 표시로 값 변화 확인이 쉬움

# -----------------------------
# 7-2. 손실 그래프
# -----------------------------
plt.subplot(1, 2, 2)
# → 1행 2열 중 두 번째 위치에 그래프 배치

plt.plot(epochs, history.history['loss'], marker='o')
# → epoch별 loss 값을 선 그래프로 그림

plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

# subplot 간격 자동 조정
plt.tight_layout()

# 최종 그래프 출력
plt.show()