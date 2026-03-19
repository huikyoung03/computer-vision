import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 1. 이미지 불러오기
image_path = 'dabo.jpg' 
img = cv.imread(image_path)

if img is None:
    print("이미지를 불러올 수 없습니다. 파일명과 경로를 확인해주세요.")
else:
    # 원본 데이터(img) 자체를 보존하기 위해 np.copy()를 사용하여 복사본 생성
    result_img = np.copy(img)

    # 에지 검출 알고리즘은 흑백 이미지에서 가장 잘 동작하므로 그레이스케일로 변환
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # 2. Canny 에지 검출
    # cv.Canny(입력 이미지, 하위 임계값(threshold1), 상위 임계값(threshold2))
    # - 픽셀의 변화량(에지 강도)이 200(상위) 이상이면 무조건 '강한 에지'로 판단
    # - 변화량이 100(하위) ~ 200 사이면, '강한 에지'와 연결되어 있을 때만 에지로 인정
    # - 변화량이 100 미만이면 에지가 아니라고 판단하여 버림
    edges = cv.Canny(gray, 100, 200)

    # 3. 확률적 허프 변환(Probabilistic Hough Transform)을 이용한 직선 검출
    # 선을 구성하는 모든 점을 계산하지 않고, 무작위로 점을 선택해 빠르게 직선을 찾는 방식
    
    rho = 1               # 거리 해상도 (픽셀 단위): 원점에서 직선까지의 거리를 1픽셀 단위로 촘촘하게 검사
    theta = np.pi / 180   # 각도 해상도 (라디안 단위): 1도(π/180 라디안) 단위로 각도를 회전하며 검사
    threshold = 50        # 직선으로 판단할 최소 교차점 수: 허프 공간에서 최소 50개의 점이 겹쳐야 선으로 인정함.
    minLineLength = 50    # 검출할 선의 최소 길이: 길이가 50픽셀보다 짧은 선은 무시 (잡음 제거 효과)
    maxLineGap = 10       # 선 위에 있다고 간주할 최대 픽셀 간격: 끊겨 있는 선이라도 간격이 10픽셀 이하면 하나의 선으로 이어줌

    # edges 이미지에서 위에서 설정한 조건들을 만족하는 선분들의 양 끝점 좌표를 반환
    lines = cv.HoughLinesP(edges, rho, theta, threshold, minLineLength=minLineLength, maxLineGap=maxLineGap)

    # 4. 검출된 직선 그리기
    # 검출된 선이 하나라도 있는지(None이 아닌지) 먼저 확인
    if lines is not None:
        for line in lines:
            # line[0]에는 선분의 시작점(x1, y1)과 끝점(x2, y2) 좌표가 들어있음
            x1, y1, x2, y2 = line[0]
            
            # cv.line(그릴 이미지, 시작점, 끝점, 색상, 두께)
            cv.line(result_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # 5. 시각화 (Matplotlib)
    # Matplotlib 출력을 위해 BGR 색상 공간을 RGB 색상 공간으로 변환
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    result_img_rgb = cv.cvtColor(result_img, cv.COLOR_BGR2RGB)

    # 그래프 전체 크기 설정 (가로 12, 세로 6)
    plt.figure(figsize=(12, 6))

    # 원본 이미지 시각화
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title('Original Image')
    plt.axis('off')

    # 직선이 검출된 결과 이미지 시각화
    plt.subplot(1, 2, 2)
    plt.imshow(result_img_rgb)
    plt.title('Detected Lines (Hough Transform)')
    plt.axis('off')

    plt.tight_layout() # 레이아웃 정렬
    plt.show()         # 화면에 출력