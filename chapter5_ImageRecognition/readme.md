# 1. MNIST 데이터셋을 이용한 간단한 이미지 분류기 구현

## 문제

손글씨 숫자 이미지(MNIST 데이터셋)를 이용하여 간단한 이미지 분류기를 구현하고, 학습 결과를 정확도와 손실 그래프로 시각화한다.

## 요구사항

• MNIST 데이터셋을 로드

• 데이터를 훈련 세트와 테스트 세트로 분할

• 간단한 신경망 모델을 구축

• 모델을 훈련시키고 정확도를 평가

• Matplotlib을 이용하여 학습 정확도와 손실값을 시각화

## 전체 코드 (01_MNIST.py)
```python
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
```

## 주요 코드 
```python
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
```
-> MNIST 손글씨 숫자 데이터셋을 불러옴.
학습용 데이터와 테스트용 데이터가 자동으로 분리되어 제공되며, 각 이미지는 28x28 크기의 흑백 이미지이다.

```python
x_train = x_train / 255.0
x_test = x_test / 255.0
```
-> 이미지의 픽셀 값은 원래 0 ~ 255 범위이므로, 이를 255.0으로 나누어 0 ~ 1 범위로 정규화함.
이 과정은 학습을 더 안정적이고 빠르게 만들어준다.

```python
layers.Flatten(input_shape=(28, 28))
```
-> 28x28 형태의 2차원 이미지를 1차원 벡터로 펼쳐주는 층이다.
완전연결층(Dense layer)에 입력하기 위해 사용된다. 

```python
layers.Dense(128, activation='relu')
```
-> 은닉층으로, 입력 데이터에서 중요한 특징을 학습하는 역할을 한다.
ReLU 활성화 함수를 사용하여 비선형적인 패턴도 학습할 수 있게 한다.

```python
layers.Dense(10, activation='softmax')
```
-> 출력층으로, 숫자 0부터 9까지 총 10개의 클래스를 분류하기 위해 10개의 노드를 사용한다.
Softmax 함수를 통해 각 클래스에 대한 확률값을 출력한다.

```python
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```
-> 모델의 학습 방법을 설정하는 부분이다.
Adam 옵티마이저를 사용하여 가중치를 업데이트하고, 다중 클래스 분류에 적합한 sparse categorical crossentropy 손실 함수를 사용한다.
또한 정확도를 함께 측정하도록 설정한다.

```python
history = model.fit(x_train, y_train, epochs=7, batch_size=64)
```
-> 학습 데이터를 사용하여 모델을 실제로 훈련시키는 부분이다.
전체 데이터를 7번 반복 학습하며, 한 번에 64개의 샘플씩 나누어 학습한다.
이때 각 epoch의 정확도와 손실값이 history 객체에 저장된다.

```python
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
```
-> 학습이 완료된 모델을 테스트 데이터로 평가하여 최종 손실값과 정확도를 구한다.
즉, 학습에 사용하지 않은 데이터에 대해 모델이 얼마나 잘 동작하는지 확인하는 단계이다.

## 결과 화면 

<img width="818" height="30" alt="Figure_3" src="https://github.com/user-attachments/assets/05e7bd86-c5ed-4c68-8a0a-cef8b0580569" />

---

# 2. CIFAR-10 데이터셋을 활용한 CNN 모델 구축

## 문제

CIFAR-10 데이터셋을 활용하여 합성곱 신경망(CNN)을 구축하고 이미지 분류를 수행한다. 또한 학습된 모델을 이용해 같은 폴더에 있는 dog.jpg 이미지에 대한 예측을 수행한다.


## 요구사항

• CIFAR-10 데이터셋을 로드

• 데이터 전처리(정규화 등)를 수행

• CNN 모델을 설계하고 훈련

• 모델의 성능을 평가

• 테스트 이미지(dog.jpg)에 대한 예측을 수행

## 전체 코드 (02_cifar10.py)

``` python 
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
```

## 주요 코드 

```python
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
```
-> CIFAR-10 데이터셋을 불러오는 코드이다.
32×32 크기의 컬러 이미지로 구성되어 있으며, 학습용 데이터와 테스트용 데이터가 자동으로 분리되어 제공된다.

```python
train_images, test_images = train_images / 255.0, test_images / 255.0
```
-> 이미지 픽셀 값은 원래 0 ~ 255 범위이므로, 이를 255.0으로 나누어 0 ~ 1 범위로 정규화한다.
이 과정은 학습을 더 안정적으로 만들고 수렴 속도를 높이는 데 도움이 된다.

```python
layers.Conv2D(32, (3, 3), padding='same', activation='relu')
```
-> 합성곱 층으로, 이미지의 특징(엣지, 패턴 등)을 추출한다.
padding='same'을 사용하여 입력과 출력 크기를 유지한다.

```python
layers.MaxPooling2D((2, 2))
```
-> 특징 맵의 크기를 줄여 연산량을 감소시키고, 중요한 특징만 남긴다.

```python
layers.Dropout(0.25)
```
-> 일부 뉴런을 비활성화하여 과적합을 방지한다.

```python
layers.Flatten()
```
-> 합성곱 층의 2차원 데이터를 1차원 벡터로 변환한다.

```python
layers.Dense(256, activation='relu')
```
-> 추출된 특징을 기반으로 분류를 위한 패턴을 학습한다.

```python
layers.Dense(10, activation='softmax')
```
-> 10개 클래스에 대한 확률을 출력한다.
Softmax를 사용하여 각 클래스의 확률 합이 1이 되도록 한다.


```python
loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
```
-> 정답 레이블이 정수 형태일 때 사용하는 다중 클래스 분류용 손실 함수이다.
from_logits=True는 출력층 결과가 확률값이 아니라 logits 값이라는 것을 의미한다.

```python
model.fit(
    train_images, train_labels,
    epochs=20,
    validation_data=(test_images, test_labels)
)
```
-> 학습 데이터를 사용하여 모델을 훈련하는 부분이다.
전체 학습 데이터를 20번 반복 학습하고, 각 epoch마다 테스트 데이터를 이용해 검증 성능도 함께 확인한다.

```python
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
```
-> 학습이 완료된 모델을 테스트 데이터로 평가하여 손실값과 정확도를 측정한다.
즉, 학습에 사용하지 않은 데이터에 대해 모델이 얼마나 잘 일반화되는지 확인하는 단계이다.

```python
img = image.load_img(img_path, target_size=(32, 32))
```
-> 외부 이미지 dog.jpg를 불러오고, 모델 입력 크기에 맞도록 32×32 크기로 조정한다.

```python
img_array = np.expand_dims(img_array, axis=0)
```
-> 단일 이미지에 배치 차원을 추가하여 (1, 32, 32, 3) 형태로 만든다.
TensorFlow 모델은 여러 장의 이미지를 한 번에 처리하는 형태를 기본으로 하기 때문에 이 과정이 필요하다.

```python   
score = tf.nn.softmax(predictions[0])
```
-> 출력층의 logits 값을 softmax 함수로 변환하여 각 클래스에 대한 확률값으로 바꾼다.
이후 가장 높은 확률을 가진 클래스를 최종 예측 결과로 사용한다. 가로 길이를 더한 크기(panorama_w)로 설정함.

## 결과 
<img width="763" height="186" alt="Figure_2" src="https://github.com/user-attachments/assets/0886e46b-22b8-4bc3-b94c-d1e99a74ee1e" />

