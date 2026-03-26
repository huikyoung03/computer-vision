import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 1. 두 개의 이미지를 불러옴 [cite: 52]
img1 = cv.imread('img2.jpg') 
img2 = cv.imread('img3.jpg')

# 2. SIFT 특징점 검출 [cite: 53]
sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# 3. BFMatcher와 knnMatch를 사용하여 특징점 매칭 [cite: 54]
bf = cv.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# 거리 비율이 임계값(0.7) 미만인 좋은 매칭점만 선별 [cite: 62]
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

# 매칭 결과 시각화용 이미지 생성 (Matching Result)
match_img = cv.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# 4. 호모그래피 행렬 계산 [cite: 55]
# img2를 변환하여 img1 옆에 붙이는 방향으로 계산
src_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# RANSAC을 사용하여 이상점 영향 줄임 [cite: 60]
H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

# 5. warpPerspective를 사용하여 한 이미지를 변환하여 다른 이미지와 정렬 [cite: 56]
h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]

# 출력 크기를 두 이미지를 합친 파노라마 크기 (w1+w2, max(h1,h2))로 설정 
panorama_w = w1 + w2
panorama_h = max(h1, h2)

# img2를 변환
warped_img = cv.warpPerspective(img2, H, (panorama_w, panorama_h))

# 원본 img1을 변환된 캔버스의 제자리에 덮어쓰기
warped_img[0:h1, 0:w1] = img1

# 6. 변환된 이미지와 특징점 매칭 결과를 나란히 출력 
plt.figure(figsize=(20, 8))

# 특징점 매칭 결과 (왼쪽)
plt.subplot(1, 2, 1)
plt.imshow(cv.cvtColor(match_img, cv.COLOR_BGR2RGB))
plt.title('Matching Result')
plt.axis('off')

# 호모그래피 정합 결과 (오른쪽)
plt.subplot(1, 2, 2)
plt.imshow(cv.cvtColor(warped_img, cv.COLOR_BGR2RGB))
plt.title('Warped Image (Alignment)')
plt.axis('off')

plt.tight_layout()
plt.show()