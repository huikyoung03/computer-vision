import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# 이미지 불러오기
image_path = 'edgeDetectionImage.jpg' 
img = cv.imread(image_path)

# 이미지가 정상적으로 불러와졌는지 확인
if img is None:
    print("이미지를 불러올 수 없습니다. 경로를 확인해주세요.")
else:
    # [데이터 전처리 과정]
    # Matplotlib 라이브러리는 RGB(빨강, 초록, 파랑) 순서로 색상을 해석
    # 따라서 화면에 원본을 원래 색상대로 띄우려면 cv.cvtColor를 이용해 BGR을 RGB로 변환
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # 에지(윤곽선) 검출은 색상보다 '밝기의 변화(명암)'를 찾는 작업
    # 불필요한 색상 정보를 없애고 연산 속도를 높이기 위해 흑백(Grayscale) 이미지로 변환
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # [소벨(Sobel) 에지 검출 연산]
    # cv.Sobel(입력 이미지, 출력 데이터 타입, x방향 미분, y방향 미분, 커널 크기)
    # 왜 cv.CV_64F? 
    # -> 픽셀값의 차이(미분)를 구할 때 값이 음수가 나올 수 있음.  
    #    기본 8비트(uint8)를 쓰면 음수가 모두 0으로 잘려버려 정보가 손실되므로, 마이너스 값도 보존하기 위해 실수형 사용
    sobel_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3) # 수직선(x축 방향 밝기 변화) 검출
    sobel_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3) # 수평선(y축 방향 밝기 변화) 검출

    # [에지 강도(Magnitude) 계산]
    # x방향 미분값과 y방향 미분값을 이용해 해당 픽셀의 최종적인 전체 에지 강도를 구함
    magnitude = cv.magnitude(sobel_x, sobel_y)

    # 화면에 이미지를 출력하려면 픽셀 값을 다시 0~255 사이의 8비트 정수형(uint8)으로 변환
    # cv.convertScaleAbs()는 음수 값에 절댓값을 취해주고, 데이터를 다시 uint8 타입으로 안전하게 변환
    magnitude_uint8 = cv.convertScaleAbs(magnitude)

    # [시각화 (Matplotlib)]
    # 그림판 크기 설정
    plt.figure(figsize=(12, 6))

    # 1행 2열의 화면 중 첫 번째 자리에 원본 이미지를 배치
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title('Original Image')
    plt.axis('off') # 이미지만 깔끔하게 보이도록 x축, y축 눈금을 숨김

    # 1행 2열의 화면 중 두 번째 자리에 에지 이미지를 배치
    plt.subplot(1, 2, 2)
    # 흑백(Grayscale) 이미지이므로 cmap='gray' 컬러맵을 지정해야 정상적인 흑백 톤으로 렌더링
    plt.imshow(magnitude_uint8, cmap='gray')
    plt.title('Sobel Edge Magnitude')
    plt.axis('off')

    # 여러 개의 그래프 간격이 겹치지 않게 여백을 자동으로 예쁘게 조절
    plt.tight_layout()
    # 최종적으로 설정한 그림판을 화면에 띄움
    plt.show()