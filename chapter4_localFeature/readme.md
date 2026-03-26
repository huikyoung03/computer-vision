# 1. SIFT 특징점 검출 및 결과 시각화

## 문제

주어진 이미지(mot_color70.jpg)에서 SIFT 알고리즘을 사용하여 특징점을 검출하고 크기와 방향을 포함하여 시각화한다.

## 요구사항

• cv.imread()를 사용하여 이미지를 불러옴

• cv.cvtColor()를 사용하여 그레이스케일로 변환

• cv.SIFT_create()를 사용하여 SIFT 객체를 생성하고, 특징점 개수 제한 옵션을 설정

• detectAndCompute()를 사용하여 특징점을 검출

• cv.drawKeypoints()의 DRAW_RICH_KEYPOINTS 플래그를 사용하여 특징점을 이미지에 시각화

• Matplotlib을 이용하여 원본 이미지와 특징점이 시각화된 이미지를 나란히 출력

## 전체 코드 (01_SIFT.py)
```python
import cv2 as cv
import matplotlib.pyplot as plt

# 1. 이미지 로드 (mot_color70.jpg)
image_path = 'mot_color70.jpg'
img = cv.imread(image_path)

if img is None:
    print("이미지를 불러올 수 없습니다. 경로를 확인해주세요.")
else:
    # 에지 및 특징점 검출은 흑백 이미지에서 수행하는 것이 일반적
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # 2. SIFT 객체 생성 (nfeatures로 특징점 개수 제한 가능)
    # 너무 많은 특징점이 추출되는 것을 방지하기 위해 상위 500개만 추출하도록 설정
    sift = cv.SIFT_create(nfeatures=500)

    # 3. 특징점 검출
    # detectAndCompute 함수는 특징점의 위치 정보(keypoints)와 
    # 해당 특징점 주변의 패턴 정보(descriptors)를 함께 반환
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    # 4. 특징점 시각화 (방향과 크기 표시)
    # flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS를 설정하면 
    # 특징점의 위치뿐만 아니라 크기(원)와 방향(선)도 함께 그려줌
    img_with_kp = cv.drawKeypoints(img, keypoints, None, 
                                   flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # 5. 결과 출력 (Matplotlib)
    plt.figure(figsize=(12, 6))
    
    # 첫 번째: 원본 이미지
    plt.subplot(1, 2, 1)
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    # 두 번째: 특징점이 표시된 이미지
    plt.subplot(1, 2, 2)
    plt.imshow(cv.cvtColor(img_with_kp, cv.COLOR_BGR2RGB))
    plt.title('SIFT Keypoints (Rich)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
```

## 주요 코드 
   
   sift = cv.SIFT_create(nfeatures=500)
-> 크기(Scale)와 회전(Rotation)에 불변하는 특징점을 찾기 위한 SIFT 객체를 생성. 

nfeatures를 통해 검출할 최대 특징점의 개수를 제한함.
    
    keypoints, descriptors = sift.detectAndCompute(gray, None)
-> 입력 이미지에서 특징점(keypoints)을 찾고, 그 주변 픽셀의 정보들을 요약하여 다른 이미지와 비교할 수 있는 형태인 기술자(descriptors)를 계산함.

    cv.drawKeypoints(img, keypoints, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)  
-> 검출된 특징점을 이미지 위에 그림. DRAW_RICH_KEYPOINTS 플래그는 각 특징점의 스케일(원의 크기)과 방향(원 내부의 선)을 시각적으로 나타냄.
   

## 결과 화면 


---
# 2. SIFT를 이용한 두 영상 간 특징점 매칭

## 문제

두 개의 이미지(mot_color70.jpg, mot_color83.jpg)를 입력받아 SIFT 특징점을 기반으로 공통된 부분을 찾아 매칭하고 결과를 시각화한다.


## 요구사항

• cv.imread()를 사용하여 두 개의 이미지를 불러옴

• cv.SIFT_create()를 사용하여 각 이미지의 특징점과 기술자를 추출

• cv.BFMatcher()를 사용하여 두 영상 간 특징점을 1:1로 매칭

• 추출된 매칭 결과들을 거리에 따라 정렬하여 상위 매칭점만 선별

• cv.drawMatches()를 사용하여 두 이미지 간의 매칭 결과를 선으로 연결하여 시각화

• Matplotlib을 이용하여 결과를 출력

## 전체 코드 (02_SIFT_matching.py)

``` python 
import cv2 as cv
import matplotlib.pyplot as plt

# 1. 두 개의 이미지 불러오기
img1 = cv.imread('mot_color70.jpg')
img2 = cv.imread('mot_color83.jpg')

if img1 is None or img2 is None:
    print("이미지를 불러올 수 없습니다. 경로를 확인해주세요.")
else:
    # 2. SIFT 특징점 및 기술자 추출
    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # 3. BFMatcher 생성 및 매칭
    # cv.NORM_L2: SIFT 기술자 비교에 사용하는 유클리디안 거리 측정 방식
    # crossCheck=True: 상호 매칭 검사를 통해 양방향으로 가장 가까운 쌍만 매칭으로 인정 (오매칭 감소)
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)

    # 4. 거리에 따라 정렬 (선택 사항)
    # distance 값이 작을수록 두 특징점이 매우 유사하다는 의미이므로 오름차순 정렬
    matches = sorted(matches, key=lambda x: x.distance)

    # 5. 매칭 결과 시각화
    # 상위 50개의 매칭 결과만 화면에 그려서 복잡함을 줄임
    # flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS: 짝을 찾지 못한 특징점은 그리지 않음
    res = cv.drawMatches(img1, kp1, img2, kp2, matches[:50], None, 
                         flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # 6. 결과 출력
    plt.figure(figsize=(15, 8))
    plt.imshow(cv.cvtColor(res, cv.COLOR_BGR2RGB))
    plt.title('SIFT Feature Matching')
    plt.axis('off')
    plt.show()
```

## 주요 코드 

    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
-> 전수 조사(Brute-Force) 방식을 사용하여 모든 기술자의 거리를 계산해 가장 유사한 쌍을 찾음. 

crossCheck=True를 설정하면 A에서 B로 갈 때 가장 가깝고, B에서 A로 갈 때도 가장 가까운 경우만 매칭으로 인정함.

    matches = sorted(matches, key=lambda x: x.distance)
-> distance 속성은 매칭된 두 특징점 간의 거리(유사도)를 나타냄. 

값이 작을수록 일치율이 높으므로, 오름차순으로 정렬하여 신뢰도가 높은 매칭점을 앞쪽으로 배치함.

    cv.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
-> 두 이미지를 나란히 배치하고, 매칭된 특징점 쌍을 선으로 이어줌. 

matches[:50]을 통해 정렬된 리스트 중 가장 정확한 50개만 그려 시각적인 복잡함을 덜어냄.

## 결과 화면 



---

# 3. 호모그래피를 이용한 이미지 정합 (Image Alignment)

## 문제

SIFT 특징점을 사용하여 두 이미지 간 대응점을 찾고, 이를 바탕으로 호모그래피 행렬을 계산하여 하나의 이미지 위에 정렬하여 파노라마를 생성한다.

## 요구사항

• cv.imread()를 사용하여 두 개의 이미지를 불러옴

• cv.SIFT_create()를 사용하여 특징점을 검출

• cv.BFMatcher()와 knnMatch()를 사용하여 특징점을 매칭하고, 거리 비율(Ratio Test) 임계값을 통해 좋은 매칭점만 선별

• cv.findHomography()와 cv.RANSAC을 사용하여 호모그래피 행렬을 계산

• cv.warpPerspective()를 사용하여 한 이미지를 변환하고 파노라마 캔버스에 정합

• Matplotlib을 사용하여 특징점 매칭 결과(Matching Result)와 변환된 파노라마 이미지(Warped Image)를 나란히 출력

## 전체 코드 (03_Homography.py)

```python
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 1. 두 개의 이미지를 불러옴
img1 = cv.imread('img2.jpg') 
img2 = cv.imread('img3.jpg')

if img1 is None or img2 is None:
    print("이미지를 불러올 수 없습니다.")
else:
    # 2. SIFT 특징점 검출
    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # 3. BFMatcher와 knnMatch를 사용하여 특징점 매칭
    # k=2: 각 특징점마다 가장 유사한 상위 2개의 매칭점을 찾음
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # 4. 거리 비율(Ratio Test)을 이용한 좋은 매칭점 선별
    good_matches = []
    for m, n in matches:
        # 가장 가까운 매칭점(m)의 거리가 두 번째로 가까운 매칭점(n) 거리의 70% 미만일 때만 진짜 매칭으로 인정 (Lowe's ratio test)
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # 매칭 결과 시각화용 이미지 생성 (Matching Result)
    match_img = cv.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # 5. 호모그래피 행렬 계산
    # 변환할 이미지(img2)의 특징점 좌표를 src_pts, 기준이 될 이미지(img1)의 좌표를 dst_pts로 설정
    src_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # RANSAC을 사용하여 잘못 매칭된 점들(이상점)의 영향을 배제하고 투시 변환 행렬(H)을 구함
    H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

    # 6. warpPerspective를 사용하여 변환 및 파노라마 캔버스 정렬
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # 출력 크기를 두 이미지를 합친 넉넉한 파노라마 크기 (w1+w2, max(h1,h2))로 설정 
    panorama_w = w1 + w2
    panorama_h = max(h1, h2)

    # img2를 계산된 호모그래피 행렬(H)에 따라 img1의 시점으로 변환시킴 (배경 캔버스에 배치)
    warped_img = cv.warpPerspective(img2, H, (panorama_w, panorama_h))

    # 변환된 이미지의 제자리(왼쪽 위)에 기준이 되는 원본 img1을 덮어써서 두 이미지를 이어 붙임
    warped_img[0:h1, 0:w1] = img1

    # 7. 결과 시각화
    plt.figure(figsize=(20, 8))

    # 특징점 매칭 결과 시각화
    plt.subplot(1, 2, 1)
    plt.imshow(cv.cvtColor(match_img, cv.COLOR_BGR2RGB))
    plt.title('Matching Result')
    plt.axis('off')

    # 호모그래피 파노라마 결과 시각화
    plt.subplot(1, 2, 2)
    plt.imshow(cv.cvtColor(warped_img, cv.COLOR_BGR2RGB))
    plt.title('Warped Image (Alignment)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
```


## 주요 코드 

   matches = bf.knnMatch(des1, des2, k=2)
-> 단순한 1:1 매칭이 아니라, 각 특징점에 대해 가장 유사한 이웃(Nearest Neighbor)을 k개(여기서는 2개)만큼 반환함.
    if m.distance < 0.7 * n.distance:
-> Lowe's ratio test를 적용. 

가장 가까운 매칭점(1순위)과 두 번째로 가까운 매칭점(2순위)의 거리 차이가 명확할 때(즉, 확실하게 하나만 일치할 때)만 해당 점을 채택하여 오매칭을 크게 줄임. 
    
    H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
-> 선별된 좋은 매칭점들을 이용해 두 이미지 평면 간의 원근 변환 관계(호모그래피 행렬, 3x3)를 계산함. 

    cv.RANSAC을 사용하여 여전히 남아있는 오매칭(Outlier)의 영향을 무시하고 안정적인 행렬을 계산함. 
    
    warped_img = cv.warpPerspective(img2, H, (panorama_w, panorama_h))
-> 도출된 호모그래피 행렬 H를 이용해 입력 이미지(img2)의 모든 픽셀 좌표를 새로운 투시 공간으로 이동(변환)시킴. 

이 때 캔버스는 두 이미지가 겹쳐질 수 있도록 가로 길이를 더한 크기(panorama_w)로 설정함.

## 결과 

