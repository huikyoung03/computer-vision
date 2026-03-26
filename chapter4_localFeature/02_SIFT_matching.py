import cv2 as cv
import matplotlib.pyplot as plt

# 1. 두 개의 이미지 불러오기 [cite: 34]
img1 = cv.imread('mot_color70.jpg')
img2 = cv.imread('mot_color83.jpg')

# 2. SIFT 특징점 추출 [cite: 38]
sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# 3. BFMatcher 생성 및 매칭 (L2 노름 사용) [cite: 39, 42]
bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
matches = bf.match(des1, des2)

# 4. 거리에 따라 정렬 (선택 사항)
matches = sorted(matches, key=lambda x: x.distance)

# 5. 매칭 결과 시각화 [cite: 40, 41]
res = cv.drawMatches(img1, kp1, img2, kp2, matches[:50], None, 
                     flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.figure(figsize=(15, 8))
plt.imshow(cv.cvtColor(res, cv.COLOR_BGR2RGB))
plt.title('SIFT Feature Matching')
plt.axis('off')
plt.show()