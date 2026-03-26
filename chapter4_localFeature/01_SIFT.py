import cv2 as cv
import matplotlib.pyplot as plt

# 1. 이미지 로드 (mot_color70.jpg)
img = cv.imread('mot_color70.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 2. SIFT 객체 생성 (nfeatures로 특징점 개수 제한 가능) [cite: 20, 25]
sift = cv.SIFT_create(nfeatures=500)

# 3. 특징점 검출 [cite: 21]
keypoints, descriptors = sift.detectAndCompute(gray, None)

# 4. 특징점 시각화 (방향과 크기 표시) [cite: 22, 26]
img_with_kp = cv.drawKeypoints(img, keypoints, None, 
                               flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# 5. 결과 출력 [cite: 23]
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv.cvtColor(img_with_kp, cv.COLOR_BGR2RGB))
plt.title('SIFT Keypoints (Rich)')
plt.axis('off')

plt.show()