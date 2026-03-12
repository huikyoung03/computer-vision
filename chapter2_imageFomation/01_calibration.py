import cv2
import numpy as np
import glob

# 체크보드 내부 코너 개수
CHECKERBOARD = (9, 6)

# 체크보드 한 칸 실제 크기 (mm)
square_size = 25.0

# 코너 정밀화 조건
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 실제 좌표 생성
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size

# 저장할 좌표
objpoints = []
imgpoints = []

# 이미지 경로
images = glob.glob("./images/calibration_images/left*.jpg")

img_size = None

# -----------------------------
# 1. 체크보드 코너 검출
# -----------------------------
if len(images) == 0:
    raise FileNotFoundError("체크보드 이미지를 찾을 수 없습니다.")

for fname in images:
    img = cv2.imread(fname)

    if img is None:
        print(f"이미지 로드 실패: {fname}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_size = gray.shape[::-1]

    # 체크보드 코너 검출
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        # 코너 정밀화
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        objpoints.append(objp)
        imgpoints.append(corners2)

        # 검출 결과 시각화
        vis = img.copy()
        cv2.drawChessboardCorners(vis, CHECKERBOARD, corners2, ret)
        cv2.imshow("Chessboard Corners", vis)
        cv2.waitKey(300)
    else:
        print(f"코너 검출 실패: {fname}")

cv2.destroyAllWindows()

if len(objpoints) == 0 or len(imgpoints) == 0:
    raise RuntimeError("유효한 체크보드 코너를 검출하지 못했습니다.")

# -----------------------------
# 2. 카메라 캘리브레이션
# -----------------------------
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    img_size,
    None,
    None
)

print("Reprojection Error:")
print(ret)

print("\nCamera Matrix K:")
print(K)

print("\nDistortion Coefficients:")
print(dist)

# -----------------------------
# 3. 왜곡 보정 시각화
# -----------------------------
sample_img = cv2.imread(images[0])

if sample_img is None:
    raise FileNotFoundError("샘플 이미지를 불러올 수 없습니다.")

undistorted = cv2.undistort(sample_img, K, dist, None, K)

result = np.hstack([sample_img, undistorted])

cv2.imshow("Original | Undistorted", result)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("calibration_result.jpg", result)
print("\n왜곡 보정 결과 저장 완료: calibration_result.jpg")