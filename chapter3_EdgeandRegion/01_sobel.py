import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# 1. cv.imread()를 사용하여 이미지를 불러옴
# 본인이 가진 이미지 파일명으로 변경하세요. (예: 'edgeDetectionImage.jpg')
image_path = 'edgeDetectionImage.jpg' 
img = cv.imread(image_path)

if img is None:
    print("이미지를 불러올 수 없습니다. 경로를 확인해주세요.")
else:
    # Matplotlib에서 원본 이미지를 올바른 색상으로 출력하기 위해 BGR을 RGB로 변환
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # 2. cv.cvtColor()를 사용하여 그레이스케일로 변환
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # 3. cv.Sobel()을 사용하여 x축과 y축 방향의 에지를 검출 (힌트: ksize는 3으로 설정)
    sobel_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
    sobel_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)

    # 4. cv.magnitude()를 사용하여 에지 강도 계산
    magnitude = cv.magnitude(sobel_x, sobel_y)

    # 힌트: cv.convertScaleAbs()를 사용하여 에지 강도 이미지를 uint8로 변환
    magnitude_uint8 = cv.convertScaleAbs(magnitude)

    # 5. Matplotlib를 사용하여 원본 이미지와 에지 강도 이미지를 나란히 시각화
    plt.figure(figsize=(12, 6))

    # 원본 이미지 시각화
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title('Original Image')
    plt.axis('off') # 축 눈금 숨기기

    # 에지 강도 이미지 시각화 (힌트: cmap='gray' 사용)
    plt.subplot(1, 2, 2)
    plt.imshow(magnitude_uint8, cmap='gray')
    plt.title('Sobel Edge Magnitude')
    plt.axis('off')

    plt.tight_layout()
    plt.show()