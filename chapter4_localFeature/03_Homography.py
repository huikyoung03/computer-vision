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