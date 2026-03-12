# 02_rotation_transformation.py
import cv2
import numpy as np

# 이미지 경로
IMAGE_PATH = "./images/rose.png"

img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {IMAGE_PATH}")

 #이미지 크기 변경
scale = 0.5                                     # 이미지 크기 축소 (원본의 50%)    
img = cv2.resize(img, None, fx=scale, fy=scale)

h, w = img.shape[:2]
center = (w // 2, h // 2)

# 회전 + 스케일
angle = 30
scale = 0.8

M = cv2.getRotationMatrix2D(center, angle, scale)

# 평행이동 추가
tx = 80   # x축 +80
ty = -40  # y축 -40
M[0, 2] += tx
M[1, 2] += ty

# 변환 적용
transformed = cv2.warpAffine(img, M, (w, h))

# 결과 비교
result = np.hstack([img, transformed])

cv2.imshow("Original | Rotated + Scaled + Translated", result)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("rotation_transformation_result.jpg", result)
print("저장 완료: rotation_transformation_result.jpg")