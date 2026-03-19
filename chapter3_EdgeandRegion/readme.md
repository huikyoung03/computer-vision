# 1. 소벨 에지 검출 및 결과 시각화 

## 문제

edgeDetectionImage를 sobel 필터를 사용하여 에지 검출한 결과를 시각화한다.

## 요구사항

• cv.imread()를 사용하여 이미지를 불러옴

• cv.cvtColor()를 사용하여 그레이스케일로 변환

• cvSobel()을 사용하여 x축(cv.CV_64F, 1, 0)과 y축(cv.CV_64F, 0,1) 방향의 에지를 검출

• cv.magnitude()를 사용하여 에지 강도 계산

• Matplotlib를 사용하여 원본 이미지와 에지 강도 이미지를 나란히 시각화

## 전체 코드 (01_sobel.py)

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


## 주요 코드 
   
    sobel_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
    sobel_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)
-> x축, y축 방향의 변화량(미분값)을 각각 계산 (음수 보존을 위해 CV_64F 사용)

    magnitude = cv.magnitude(sobel_x, sobel_y)
-> x방향 미분값과 y방향 미분값을 이용해 해당 픽셀의 최종적인 전체 에지 강도 계산

    cv.convertScaleAbs(magnitude)    
-> 음수 값에 절댓값을 취하여 데이터를 다시 uint8 타입으로 안전하게 변환

    plt.imshow(magnitude_uint8, cmap='gray')
-> Matplotlib를 사용, 흑백(Grayscale) 이미지이므로 cmap='gray' 컬러맵을 지정해야 정상적인 흑백 톤으로 렌더링 가능
        

## 결과 화면 

<img width="1200" height="600" alt="Figure_1" src="https://github.com/user-attachments/assets/361cac78-c80b-415f-8f2c-95558821ed71" />

---
# 2. 캐니 에지 및 허프 변환을 이용행 직선 검출

## 문제

dabo 이미지에서 케니 에지 검출을 사용하여 에지 맵을 생성한다

허프 변환을 사용하여 이미지에서 직선을 검출한다.


## 요구사항

• cv.Canny()를 사용하여 에지 맵 생성

• cv.HoughtLinesP()를 사용하여 직선 검출

• cv.line()을 사용하여 검출된 직선을 원본 이미지에 그림

• Matplotlib를 사용하여 원본 이미지와 직선이 그려진 이미지를 나란히 시각화


## 전체 코드 (02_canny_hough.py)

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




## 주요 코드 

    edges = cv.Canny(gray, 100, 200)
-> Canny 에지 검출 알고리즘을 사용하여 에지 맵 생성

-> 픽셀 변화량이 200 이상이면 확실한 에지, 100~200 사이면 연결된 경우만 에지로 인정

    lines = cv.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)
-> 확률적 허프 변환(Probabilistic Hough Transform)을 이용하여 edges(흑백 에지 맵)에서 최소 50개의 점이 연속되고, 끊긴 간격이 10 이하인 직선들만 추출

    cv.line(result_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
-> 검출된 직선을 원본 이미지에 그림

    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    result_img_rgb = cv.cvtColor(result_img, cv.COLOR_BGR2RGB)
->Matplotlib 출력을 위해 BGR 색상 공간을 RGB 색상 공간으로 변환

## 결과 화면 

<img width="1200" height="600" alt="Figure_2" src="https://github.com/user-attachments/assets/c6926c3c-e565-4db4-a3cd-d3256c178121" />


---

# 3. GrabCut을 이용한 대화식 영역 분할 및 객체 추출 

## 문제

GrabCut 알고리즘을 사용하여 이미지에서 특정 객체를 분할하고 추출한다.

## 요구사항

• cv.grabCut()를 사용하여 대화식 분할을 수행

• 초기 사각형 영역은 (x, y, width, height) 형식으로 설정

• 마스크를 사용하여 원본 이미지에서 배경을 제거

• matplotlib를 사용하여 원본 이미지, 마스크 이미지, 배경 제거 이미지 세 개를 나란히 시각화

## 전체 코드 (03_grabcut.py)

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


## 주요 코드 

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
-> GrabCut 알고리즘 내부에서 배경과 전경의 색상 분포를 학습하고 저장하기 위해 사용되는 모델 배열 생성

    rect = (x, y, width, height)
-> 사각형 영역 설정 (x좌표, y좌표, 가로 너비, 세로 높이)

-> 추출하려는 객체를 완전히 포함하면서, 객체에 최대한 가깝게 잡아야 함. 

  배경이 너무 많이 들어가면 알고리즘이 객체 색상 모델을 학습할 때 혼란을 느껴 배경까지 객체로 오려버릴 수 있음

    cv.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)
-> GrabCut 알고리즘을 사용하여 대화식 분할을 수행

  5: 알고리즘을 5번 반복해서 점점 더 정교하게 영역을 분할

  cv.GC_INIT_WITH_RECT: 우리가 제공한 '사각형(rect)' 정보를 바탕으로 분할을 시작

이 과정이 끝나면 mask 안에는 0~3까지의 값이 들어감.

- 0: 확실한 배경 (cv.GC_BGD) -> 사각형 밖

- 1: 확실한 전경/객체 (cv.GC_FGD) -> 사용자가 붓으로 칠했을 때만 발생

- 2: 아마도 배경 (cv.GC_PR_BGD) -> 사각형 안쪽에서 배경으로 분류된 곳

- 3: 아마도 전경/객체 (cv.GC_PR_FGD) -> 사각형 안쪽에서 객체로 분류된 곳

    mask2 = np.where((mask == cv.GC_BGD) | (mask == cv.GC_PR_BGD), 0, 1).astype('uint8')
  
-> np.where()를 이용해 조건에 맞는 값을 변경

마스크 값이 0(확실 배경)이거나 2(아마도 배경)라면 0으로 만들고, 그 외(1: 확실 객체, 3: 아마도 객체)는 1로 만듦

    result_img = img * mask2[:, :, np.newaxis]
-> 원본 이미지에 마스크를 곱해서 배경 제거
np.newaxis를 사용해 mask2를 3차원으로 늘려준 뒤 원본과 곱함.(2차원과 3차원을 바로 곱할 수 없기 때문)
배경 부분은 0을 곱하게 되므로 까맣게 지워지고, 객체 부분은 1을 곱하므로 원래 색상이 남음.


## 결과 
<img width="1500" height="500" alt="Figure_3" src="https://github.com/user-attachments/assets/753e310e-168b-4a0a-812b-9e46a5ddcad9" />

