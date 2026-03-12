import cv2
import numpy as np
import glob
from pathlib import Path

# 출력 폴더 생성
output_dir = Path("./outputs/calibration")
output_dir.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------
# 1. 체크보드 설정
# -------------------------------------------------------

# 체크보드 내부 코너 개수
# (가로 코너 수, 세로 코너 수)
# 실제 체크보드의 흰/검 칸 개수가 아니라 "코너 개수"임
CHECKERBOARD = (9, 6)

# 체크보드 한 칸의 실제 크기 (단위: mm)
# 실제 세계 좌표 계산에 사용됨
square_size = 25.0

# 코너 위치를 더 정확하게 보정하기 위한 종료 조건
# (최대 반복 횟수 30, 정확도 0.001)
criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)

# -------------------------------------------------------
# 2. 실제 세계 좌표 생성 (Object Points)
# -------------------------------------------------------

# 체크보드 코너 개수만큼 3차원 좌표 배열 생성
# (x, y, z) 좌표
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)

# x,y 좌표를 격자 형태로 생성
objp[:, :2] = np.mgrid[
    0:CHECKERBOARD[0],
    0:CHECKERBOARD[1]
].T.reshape(-1, 2)

# 실제 체크보드 크기(mm) 반영
objp *= square_size

# -------------------------------------------------------
# 3. 좌표 저장 리스트
# -------------------------------------------------------

# 실제 세계 좌표 (3D)
objpoints = []

# 이미지 좌표 (2D)
imgpoints = []

# -------------------------------------------------------
# 4. 캘리브레이션 이미지 경로
# -------------------------------------------------------

# 체크보드 이미지 목록 불러오기
# 예: left01.jpg, left02.jpg ...
images = glob.glob("./images/calibration_images/left*.jpg")

# 이미지 크기 저장 변수
img_size = None

# -------------------------------------------------------
# 5. 체크보드 코너 검출
# -------------------------------------------------------

# 이미지가 없는 경우 예외 처리
if len(images) == 0:
    raise FileNotFoundError("체크보드 이미지를 찾을 수 없습니다.")

# 각 이미지에 대해 코너 검출 수행
for fname in images:

    # 이미지 읽기
    img = cv2.imread(fname)

    if img is None:
        print(f"이미지 로드 실패: {fname}")
        continue

    # grayscale 변환
    # 코너 검출은 grayscale 이미지에서 수행
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 이미지 크기 저장
    img_size = gray.shape[::-1]

    # 체크보드 코너 검출
    ret, corners = cv2.findChessboardCorners(
        gray,
        CHECKERBOARD,
        None
    )

    # 코너 검출 성공 시
    if ret:

        # 코너 위치 정밀화 (subpixel refinement)
        corners2 = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            criteria
        )

        # 실제 좌표 저장
        objpoints.append(objp)

        # 이미지 좌표 저장
        imgpoints.append(corners2)

        # 코너 검출 결과 시각화
        vis = img.copy()

        cv2.drawChessboardCorners(
            vis,
            CHECKERBOARD,
            corners2,
            ret
        )

        cv2.imshow("Chessboard Corners", vis)

        # 0.3초 동안 표시
        cv2.waitKey(300)

    else:
        print(f"코너 검출 실패: {fname}")

cv2.destroyAllWindows()

# 코너 검출이 하나도 되지 않았을 경우 예외 처리
if len(objpoints) == 0 or len(imgpoints) == 0:
    raise RuntimeError("유효한 체크보드 코너를 검출하지 못했습니다.")

# -------------------------------------------------------
# 6. 카메라 캘리브레이션
# -------------------------------------------------------

# calibrateCamera 함수
# 입력:
#   objpoints → 실제 세계 좌표
#   imgpoints → 이미지 좌표
#   img_size  → 이미지 크기
#
# 출력:
#   ret   → reprojection error
#   K     → 카메라 내부 행렬 (Camera Matrix)
#   dist  → 왜곡 계수
#   rvecs → 회전 벡터
#   tvecs → 이동 벡터

ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    img_size,
    None,
    None
)

# 결과 출력
print("Reprojection Error:")
print(ret)

print("\nCamera Matrix K:")
print(K)

print("\nDistortion Coefficients:")
print(dist)

# -------------------------------------------------------
# 7. 왜곡 보정 (Undistortion)
# -------------------------------------------------------

# 테스트용 이미지 불러오기
sample_img = cv2.imread(images[0])

if sample_img is None:
    raise FileNotFoundError("샘플 이미지를 불러올 수 없습니다.")

# 카메라 왜곡 보정
undistorted = cv2.undistort(
    sample_img,
    K,
    dist,
    None,
    K
)

# -------------------------------------------------------
# 8. 원본 vs 보정 결과 비교
# -------------------------------------------------------

# 두 이미지를 좌우로 붙여 비교
result = np.hstack([
    sample_img,
    undistorted
])

# 결과 화면 출력
cv2.imshow("Original | Undistorted", result)

cv2.waitKey(0)
cv2.destroyAllWindows()

# -------------------------------------------------------
# 9. 결과 이미지 저장
# -------------------------------------------------------

cv2.imwrite("./outputs/calibration/calibration_result.jpg", result)

print("\n왜곡 보정 결과 저장 완료: calibration_result.jpg")