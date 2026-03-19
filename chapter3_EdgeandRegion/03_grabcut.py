import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 1. 커피 컵 이미지를 불러옵니다. (실제 파일 경로로 변경해주세요)
image_path = 'coffee_cup.jpg' 
img = cv.imread(image_path)

if img is None:
    print("이미지를 불러올 수 없습니다. 경로를 확인해주세요.")
else:
    # 원본 이미지의 복사본 및 Matplotlib 출력을 위한 RGB 변환
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    
    # 마스크 초기화 (원본 이미지와 동일한 크기의 흑백 이미지, 초기값 0)
    mask = np.zeros(img.shape[:2], np.uint8)

    # 힌트: bgdModel과 fgdModel은 np.zeros((1, 65), np.float64)로 초기화
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # 2. 초기 사각형 영역 설정 (x, y, width, height)
    # 주의: 이 값은 가지고 계신 커피 컵 이미지에 맞게 수정하셔야 합니다!
    # 객체(커피 컵)가 포함되는 대략적인 사각형 영역을 지정하세요.
    rect = (100, 100, 1000, 800) # 예시 좌표입니다.

    # 3. cv.grabCut()을 사용하여 대화식 분할 수행 (반복 횟수는 5회로 설정)
    cv.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)

    # 4. np.where()를 사용하여 마스크 값을 0 또는 1로 변경
    # 힌트: 마스크 값은 cv.GC_BGD(0), cv.GC_FGD(1), cv.GC_PR_BGD(2), cv.GC_PR_FGD(3)를 사용
    # 확실한 배경(0)이거나 아마도 배경(2)인 부분은 0으로, 나머지는 1로 만듭니다.
    mask2 = np.where((mask == cv.GC_BGD) | (mask == cv.GC_PR_BGD), 0, 1).astype('uint8')

    # 5. 원본 이미지에 곱하여 배경을 제거 (마스크 차원을 이미지와 맞추기 위해 np.newaxis 사용)
    result_img = img * mask2[:, :, np.newaxis]
    result_img_rgb = cv.cvtColor(result_img, cv.COLOR_BGR2RGB)

    # 시각화용 마스크 이미지 (0과 1로 된 값을 0과 255로 늘려 화면에 보이게 함)
    mask_display = mask2 * 255

    # 6. matplotlib를 사용하여 세 개의 이미지를 나란히 시각화
    plt.figure(figsize=(15, 5))

    # 원본 이미지
    plt.subplot(1, 3, 1)
    plt.imshow(img_rgb)
    # 사각형 영역 표시 (빨간색)
    x, y, w, h = rect
    plt.gca().add_patch(plt.Rectangle((x, y), w, h, edgecolor='r', facecolor='none', lw=2))
    plt.title('Original Image with Rect')
    plt.axis('off')

    # 마스크 이미지
    plt.subplot(1, 3, 2)
    plt.imshow(mask_display, cmap='gray')
    plt.title('GrabCut Mask')
    plt.axis('off')

    # 배경 제거 이미지
    plt.subplot(1, 3, 3)
    plt.imshow(result_img_rgb)
    plt.title('Background Removed')
    plt.axis('off')

    plt.tight_layout()
    plt.show()