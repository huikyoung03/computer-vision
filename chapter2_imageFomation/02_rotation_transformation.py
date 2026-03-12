import cv2
import numpy as np
from pathlib import Path

# 출력 폴더 생성
output_dir = Path("./outputs/rotation_transformation")
output_dir.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------
# 1. 이미지 불러오기
# -------------------------------------------------------

# 입력 이미지 경로
IMAGE_PATH = "./images/rose.png"

# OpenCV로 이미지 읽기 (BGR 형식)
img = cv2.imread(IMAGE_PATH)

# 이미지가 정상적으로 로드되지 않았을 경우 예외 처리
if img is None:
    raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {IMAGE_PATH}")

# -------------------------------------------------------
# 2. 이미지 크기 축소
# -------------------------------------------------------

# 이미지 크기 축소 비율 설정
scale = 0.5   # 원본 이미지의 50% 크기로 축소

# cv2.resize를 이용한 이미지 크기 변경
# fx, fy : 가로 / 세로 방향 스케일 비율
img = cv2.resize(img, None, fx=scale, fy=scale)

# -------------------------------------------------------
# 3. 회전 중심 좌표 계산
# -------------------------------------------------------

# 이미지의 높이(height)와 너비(width) 추출
h, w = img.shape[:2]

# 회전 기준점 설정 (이미지 중심)
center = (w // 2, h // 2)

# -------------------------------------------------------
# 4. 회전 + 스케일 변환 행렬 생성
# -------------------------------------------------------

# 회전 각도 설정 (단위: degree)
angle = 30

# 회전과 동시에 적용할 스케일
scale = 0.8

# 회전 + 스케일을 위한 2x3 affine 변환 행렬 생성
# cv2.getRotationMatrix2D(center, angle, scale)
# center : 회전 중심
# angle  : 회전 각도
# scale  : 확대/축소 비율
M = cv2.getRotationMatrix2D(center, angle, scale)

# -------------------------------------------------------
# 5. 평행 이동 (Translation) 추가
# -------------------------------------------------------

# x축 방향 이동량 (오른쪽 이동)
tx = 80

# y축 방향 이동량 (위쪽 이동)
ty = -40

# 변환 행렬의 마지막 열은 평행 이동을 담당
# M = [ a  b  tx ]
#     [ c  d  ty ]
M[0, 2] += tx
M[1, 2] += ty

# -------------------------------------------------------
# 6. Affine 변환 적용
# -------------------------------------------------------

# cv2.warpAffine()
# img : 입력 이미지
# M   : 변환 행렬
# (w, h) : 출력 이미지 크기
transformed = cv2.warpAffine(img, M, (w, h))

# -------------------------------------------------------
# 7. 원본 이미지와 변환 이미지 비교
# -------------------------------------------------------

# 두 이미지를 좌우로 붙여서 비교
result = np.hstack([img, transformed])

# -------------------------------------------------------
# 8. 결과 출력
# -------------------------------------------------------

# 결과 이미지 화면에 출력
cv2.imshow("Original | Rotated + Scaled + Translated", result)

# 키 입력이 있을 때까지 대기
cv2.waitKey(0)

# 모든 OpenCV 창 닫기
cv2.destroyAllWindows()

# -------------------------------------------------------
# 9. 결과 이미지 저장
# -------------------------------------------------------

# 결과 이미지 파일로 저장
cv2.imwrite("./outputs/rotation_transformation/rotation_transformation_result.jpg", result)

print("저장 완료: rotation_transformation_result.jpg")