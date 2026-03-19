import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 1. 이미지 불러오기
image_path = 'coffee_cup.jpg' 
img = cv.imread(image_path)

if img is None:
    print("이미지를 불러올 수 없습니다. 경로를 확인해주세요.")
else:
    # Matplotlib 출력을 위해 BGR 색상을 RGB 색상으로 변환
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    
    # [GrabCut을 위한 준비 작업]
    # mask: 원본 이미지와 똑같은 가로x세로 크기를 가진 빈 도화지(2차원 배열)를 만듦
    # 이 마스크는 GrabCut 알고리즘이 실행되면서 픽셀이 배경인지 객체인지 기록하는 용도로 씀.
    mask = np.zeros(img.shape[:2], np.uint8)

    # bgdModel, fgdModel: GrabCut 알고리즘 내부에서 배경(bgd)과 전경/객체(fgd)의 
    # 색상 분포를 학습하고 저장하기 위해 사용
    # 크기는 항상 (1, 65)이고 타입은 float64로 지정해야 합니다. (OpenCV 규칙)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # 2. 초기 사각형 영역 설정 (x좌표, y좌표, 가로 너비, 세로 높이)
    # GrabCut 알고리즘에게 "이 사각형 밖은 무조건 배경이고, 이 안쪽에 내가 찾고 싶은 객체가 있어!"라고 알려주는 힌트
    rect = (100, 100, 1000, 800) 

    # 3. GrabCut 알고리즘 실행
    # cv.grabCut(입력 이미지, 마스크, 사각형 영역, 배경 모델, 전경 모델, 반복 횟수, 초기화 방식)
    # - 5: 알고리즘을 5번 반복해서 점점 더 정교하게 영역을 분할
    # - cv.GC_INIT_WITH_RECT: 우리가 제공한 '사각형(rect)' 정보를 바탕으로 분할을 시작
    cv.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)

    # [마스크 변환 및 배경 제거]
    # GrabCut이 끝난 후 mask 안에는 0~3까지의 값이 들어갑니다.
    # 0: 확실한 배경 (cv.GC_BGD)
    # 1: 확실한 전경/객체 (cv.GC_FGD)
    # 2: 아마도 배경 (cv.GC_PR_BGD)
    # 3: 아마도 전경/객체 (cv.GC_PR_FGD)
    
    # 4. np.where()를 이용해 조건에 맞는 값을 변경
    # 마스크 값이 0(확실한 배경)이거나 2(아마도 배경)라면 0으로 만들고, 그 외(객체)는 1로 만듬
    # => 0과 1로만 이루어진 완전한 이진 마스크(Binary Mask)로 만드는 과정
    mask2 = np.where((mask == cv.GC_BGD) | (mask == cv.GC_PR_BGD), 0, 1).astype('uint8')

    # 5. 원본 이미지에 마스크를 곱해서 배경 제거
    # mask2는 흑백(2차원)이고 원본 img는 컬러(3차원)라서 직접 곱할 수 없음
    # np.newaxis를 사용해 mask2를 3차원으로 늘려준 뒤 원본과 곱함.
    # -> 배경 부분은 0을 곱하게 되므로 까맣게 지워지고, 객체 부분은 1을 곱하므로 원래 색상이 남음
    result_img = img * mask2[:, :, np.newaxis]
    result_img_rgb = cv.cvtColor(result_img, cv.COLOR_BGR2RGB)

    # 시각화용 마스크 이미지 만들기
    # 화면에 출력할 때 0과 1은 너무 어두워서 사람 눈에 안 보이기 때문에 1(객체)에 255를 곱해서 완전한 흰색(255)으로 만들어줍니다.
    mask_display = mask2 * 255

    # 6. Matplotlib를 사용하여 세 개의 이미지를 나란히 시각화
    plt.figure(figsize=(15, 5))

    # 첫 번째: 사각형 영역이 표시된 원본 이미지
    plt.subplot(1, 3, 1)
    plt.imshow(img_rgb)
    x, y, w, h = rect
    # 원본 이미지 위에 지정했던 초기 사각형 영역을 빨간색(edgecolor='r') 테두리로 그려줍니다.
    plt.gca().add_patch(plt.Rectangle((x, y), w, h, edgecolor='r', facecolor='none', lw=2))
    plt.title('Original Image with Rect')
    plt.axis('off')

    # 두 번째: 추출된 마스크 이미지 (흑백)
    plt.subplot(1, 3, 2)
    plt.imshow(mask_display, cmap='gray')
    plt.title('GrabCut Mask')
    plt.axis('off')

    # 세 번째: 최종적으로 배경이 제거된 이미지
    plt.subplot(1, 3, 3)
    plt.imshow(result_img_rgb)
    plt.title('Background Removed')
    plt.axis('off')

    plt.tight_layout()
    plt.show()