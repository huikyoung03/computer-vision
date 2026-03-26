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
img = cv.imread('mot_color70.jpg')  
# → 이미지를 BGR 컬러 형식으로 읽어옴 (OpenCV 기본 포맷)
# → 실패 시 None 반환 (경로 확인 필요)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  
# → 컬러 이미지를 그레이스케일(단일 채널)로 변환
# → SIFT는 intensity 기반이므로 grayscale 입력 사용

# 2. SIFT 객체 생성 (특징점 최대 개수 제한)
sift = cv.SIFT_create(nfeatures=3000)  
# → SIFT(Scale-Invariant Feature Transform) 알고리즘 생성
# → nfeatures=3000 : 검출할 특징점(keypoint)의 최대 개수 제한
# → scale / rotation 변화에 강인한 특징점 검출 알고리즘

# 3. 특징점 검출 + 디스크립터 생성
keypoints, descriptors = sift.detectAndCompute(gray, None)  
# → detect: 특징점 위치(keypoint) 찾기
# → compute: 각 특징점에 대한 descriptor(128차원 벡터) 생성
# → keypoints: 위치(x, y), 크기(scale), 방향(angle) 정보 포함 객체 리스트
# → descriptors: (N x 128) numpy 배열 (각 keypoint의 특징 벡터)

# 4. 특징점 시각화 (크기 + 방향까지 표시)
img_with_kp = cv.drawKeypoints(
    img,                  # 원본 이미지
    keypoints,            # 검출된 특징점 리스트
    None,                 # 출력 이미지 (None이면 새로 생성)
    flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)
# → DRAW_RICH_KEYPOINTS 옵션:
#    - 원으로 keypoint의 scale(크기) 표시
#    - 방향(angle)을 선으로 표시
# → 단순 점이 아니라 특징의 구조까지 시각화됨

# 5. 결과 출력 (matplotlib 사용)
plt.figure(figsize=(12, 6))  
# → 전체 figure 크기 설정

plt.subplot(1, 2, 1)  
# → 1행 2열 중 첫 번째 subplot

plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))  
# → OpenCV(BGR) → matplotlib(RGB)로 색상 변환
plt.title('Original Image')  
plt.axis('off')  
# → 축 제거 (이미지 시각화용)

plt.subplot(1, 2, 2)  
# → 두 번째 subplot

plt.imshow(cv.cvtColor(img_with_kp, cv.COLOR_BGR2RGB))  
# → 특징점이 표시된 이미지 출력
plt.title('SIFT Keypoints (Rich)')  
plt.axis('off')

plt.show()  
# → 화면에 결과 출력
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

<img width="1200" height="600" alt="Figure_1" src="https://github.com/user-attachments/assets/0c59fc07-210b-4798-a4b9-62da62196ff6" />

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
# → 첫 번째 이미지 로드 (BGR 형식)
# → 특징점 기준이 되는 기준 이미지

img2 = cv.imread('mot_color83.jpg')  
# → 두 번째 이미지 로드
# → 비교 대상 이미지 (다른 시점/각도일 가능성 있음)


# 2. SIFT 특징점 추출
sift = cv.SIFT_create()  
# → SIFT 객체 생성 (scale, rotation invariant 특징 추출)

kp1, des1 = sift.detectAndCompute(img1, None)  
# → img1에서 특징점 검출 + descriptor 생성
# → kp1: keypoint 리스트 (위치, 크기, 방향 포함)
# → des1: 각 keypoint의 128차원 특징 벡터

kp2, des2 = sift.detectAndCompute(img2, None)  
# → img2에서도 동일하게 특징점 + descriptor 생성


# 3. BFMatcher 생성 및 매칭
bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)  
# → Brute-Force Matcher 생성
# → NORM_L2: SIFT descriptor는 float 기반 → 유클리드 거리 사용
# → crossCheck=True:
#    - A → B 매칭 + B → A 매칭이 서로 일치할 때만 인정
#    - 더 정확하지만 매칭 수는 줄어듦 (정밀도 ↑, recall ↓)

matches = bf.match(des1, des2)  
# → 모든 descriptor 쌍을 비교해서 가장 가까운 매칭 찾음
# → 결과: DMatch 객체 리스트
#    - queryIdx: img1 descriptor index
#    - trainIdx: img2 descriptor index
#    - distance: 두 descriptor 간 거리 (작을수록 유사)


# 4. 거리 기준으로 정렬 (좋은 매칭 우선)
matches = sorted(matches, key=lambda x: x.distance)  
# → distance가 작은 순서대로 정렬
# → 가장 유사한 특징점 매칭이 앞쪽에 위치


# 5. 매칭 결과 시각화
res = cv.drawMatches(
    img1, kp1,           # 첫 번째 이미지 + keypoints
    img2, kp2,           # 두 번째 이미지 + keypoints
    matches[:50],        # 상위 50개 매칭만 시각화
    None,
    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)
# → 두 이미지를 좌우로 붙이고 매칭된 점을 선으로 연결
# → NOT_DRAW_SINGLE_POINTS:
#    - 매칭되지 않은 keypoint는 표시하지 않음
# → 결과:
#    - 선 = 대응되는 특징점 쌍 (correspondence)


# 6. matplotlib으로 결과 출력
plt.figure(figsize=(15, 8))  
# → 출력 이미지 크기 설정

plt.imshow(cv.cvtColor(res, cv.COLOR_BGR2RGB))  
# → OpenCV(BGR) → matplotlib(RGB) 변환

plt.title('SIFT Feature Matching')  
plt.axis('off')  

plt.show()  
# → 최종 결과 출력
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

<img width="1500" height="800" alt="Figure_2" src="https://github.com/user-attachments/assets/89ae98d4-2fae-4ebf-8035-ad4c5bfa9519" />


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
# → 기준이 되는 첫 번째 이미지
# → 최종 파노라마에서 왼쪽 기준 영상처럼 사용됨

img2 = cv.imread('img3.jpg')  
# → 정합(alignment) 대상이 되는 두 번째 이미지
# → 호모그래피를 이용해 img1 좌표계로 변환할 예정


# 2. SIFT 특징점 검출
sift = cv.SIFT_create()  
# → SIFT 객체 생성
# → 크기(scale), 회전(rotation) 변화에 강인한 특징점 검출기

kp1, des1 = sift.detectAndCompute(img1, None)  
# → img1에서 keypoint와 descriptor 추출
# → kp1: 특징점 위치, 크기, 방향 정보
# → des1: 각 특징점에 대한 128차원 descriptor

kp2, des2 = sift.detectAndCompute(img2, None)  
# → img2에서도 동일하게 특징점과 descriptor 추출


# 3. BFMatcher와 knnMatch를 사용하여 특징점 매칭
bf = cv.BFMatcher()  
# → Brute-Force Matcher 생성
# → 각 descriptor를 전수 비교하여 가장 가까운 대응점 탐색
# → 기본적으로 SIFT에는 L2 distance가 사용됨

matches = bf.knnMatch(des1, des2, k=2)  
# → 각 des1의 특징점에 대해 img2에서 가장 가까운 2개 이웃을 찾음
# → k=2로 두 개를 찾는 이유:
#    Lowe's ratio test를 적용하기 위해서임
# → 결과는 [[m, n], [m, n], ...] 형태
#    m: 가장 가까운 매칭
#    n: 두 번째로 가까운 매칭


# 거리 비율이 임계값(0.7) 미만인 좋은 매칭점만 선별
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)
# → Lowe's ratio test
# → 첫 번째 후보(m)가 두 번째 후보(n)보다 충분히 더 가깝다면
#    '구분 가능한 좋은 매칭'으로 판단
# → 잘못된 매칭(outlier)을 줄이는 데 매우 중요
# → 0.7은 비교적 엄격한 기준


# 매칭 결과 시각화용 이미지 생성
match_img = cv.drawMatches(
    img1, kp1,
    img2, kp2,
    good_matches,
    None,
    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)
# → 두 이미지를 좌우로 붙인 뒤
#    good_matches에 해당하는 특징점 쌍만 선으로 연결하여 표시
# → 매칭되지 않은 단일 keypoint는 그리지 않음


# 4. 호모그래피 행렬 계산
# img2를 변환하여 img1 옆에 붙이는 방향으로 계산

src_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
# → 원본 좌표(source points): img2의 매칭점들
# → trainIdx는 des2(img2)의 descriptor 인덱스
# → 즉, "변환할 쪽 이미지"의 좌표

dst_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
# → 목적 좌표(destination points): img1의 매칭점들
# → queryIdx는 des1(img1)의 descriptor 인덱스
# → 즉, "기준이 되는 이미지"의 좌표

# RANSAC을 사용하여 이상점 영향 줄임
H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
# → img2 좌표를 img1 좌표계로 사상하는 호모그래피 행렬 H 계산
# → cv.RANSAC:
#    잘못된 매칭점(outlier)이 섞여 있어도
#    일관된 대응 관계만 이용해 강건하게 H를 추정
# → 5.0:
#    reprojection error threshold
#    이 값보다 오차가 큰 대응점은 outlier로 판단
# → mask:
#    어떤 매칭점이 inlier인지 표시하는 배열


# 5. warpPerspective를 사용하여 한 이미지를 변환하여 다른 이미지와 정렬
h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]
# → 각 이미지의 높이(h), 너비(w) 추출

# 출력 크기를 두 이미지를 합친 파노라마 크기 (w1+w2, max(h1,h2))로 설정
panorama_w = w1 + w2
panorama_h = max(h1, h2)
# → 단순히 두 이미지를 좌우로 이어붙일 수 있도록 넉넉한 캔버스 크기 설정
# → 실제 파노라마에서는 더 정교하게 bounding box를 계산하기도 함

# img2를 img1 좌표계로 변환
warped_img = cv.warpPerspective(img2, H, (panorama_w, panorama_h))
# → 호모그래피 H를 사용해 img2를 투영 변환
# → 결과적으로 img2가 img1과 같은 평면상에 정렬됨

# 원본 img1을 변환된 캔버스의 제자리에 덮어쓰기
warped_img[0:h1, 0:w1] = img1
# → 기준 이미지 img1을 왼쪽 상단에 그대로 배치
# → 현재는 단순 덮어쓰기 방식이라
#    겹치는 영역의 블렌딩은 수행하지 않음
# → 고급 파노라마에서는 feathering, multiband blending 등을 사용


# 6. 변환된 이미지와 특징점 매칭 결과를 나란히 출력
plt.figure(figsize=(20, 8))

# 특징점 매칭 결과 (왼쪽)
plt.subplot(1, 2, 1)
plt.imshow(cv.cvtColor(match_img, cv.COLOR_BGR2RGB))
# → OpenCV는 BGR, matplotlib은 RGB를 사용하므로 색상 변환 필요
plt.title('Matching Result')
plt.axis('off')

# 호모그래피 정합 결과 (오른쪽)
plt.subplot(1, 2, 2)
plt.imshow(cv.cvtColor(warped_img, cv.COLOR_BGR2RGB))
plt.title('Warped Image (Alignment)')
plt.axis('off')

plt.tight_layout()
# → subplot 간격 자동 조정

plt.show()
# → 최종 결과 화면 출력
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
<img width="1709" height="800" alt="Figure_3" src="https://github.com/user-attachments/assets/207f7049-c50a-4db6-b17b-18298df4c898" />
<img width="1709" height="800" alt="Figure_4" src="https://github.com/user-attachments/assets/04230eb7-4f78-4a4f-8f37-8398053458c7" />


