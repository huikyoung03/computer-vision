import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 1. dabo 이미지를 불러옵니다. (실제 파일 경로로 변경해주세요)
image_path = 'dabo.jpg' 
img = cv.imread(image_path)

if img is None:
    print("이미지를 불러올 수 없습니다. 파일명과 경로를 확인해주세요.")
else:
    # 직선을 그릴 원본 이미지의 복사본 생성
    result_img = np.copy(img)

    # Canny 에지 검출을 위해 그레이스케일로 변환하는 것이 일반적입니다.
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # 2. cv.Canny()를 사용하여 에지 맵 생성
    # 힌트: threshold1=100, threshold2=200 설정
    edges = cv.Canny(gray, 100, 200)

    # 3. cv.HoughLinesP()를 사용하여 직선 검출
    # 힌트: 파라미터 조정. (아래는 일반적인 초기 설정값입니다. 이미지에 맞게 튜닝하세요)
    rho = 1               # 거리 해상도 (픽셀)
    theta = np.pi / 180   # 각도 해상도 (라디안)
    threshold = 50        # 직선으로 판단할 최소 교차점 수
    minLineLength = 50    # 검출할 선의 최소 길이
    maxLineGap = 10       # 선 위에 있다고 간주할 최대 픽셀 간격

    lines = cv.HoughLinesP(edges, rho, theta, threshold, minLineLength=minLineLength, maxLineGap=maxLineGap)

    # 4. cv.line()을 사용하여 검출된 직선을 원본 이미지 복사본에 그림
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # 힌트: 색상은 (0, 0, 255) (BGR 기준 빨간색), 두께는 2
            cv.line(result_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # 5. Matplotlib를 사용하여 원본 이미지와 직선이 그려진 이미지를 나란히 시각화
    # Matplotlib 출력을 위해 BGR에서 RGB로 변환
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    result_img_rgb = cv.cvtColor(result_img, cv.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 6))

    # 원본 이미지 시각화
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title('Original Image')
    plt.axis('off')

    # 직선이 검출된 이미지 시각화
    plt.subplot(1, 2, 2)
    plt.imshow(result_img_rgb)
    plt.title('Detected Lines (Hough Transform)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()