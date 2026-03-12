import cv2
import numpy as np
from pathlib import Path

# -------------------------------------------------------
# 1. 출력 폴더 생성
# -------------------------------------------------------
# 결과 이미지들을 저장할 폴더 생성
# 존재하지 않으면 새로 만들고, 이미 있으면 그대로 사용
output_dir = Path("./outputs/depth")
output_dir.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------
# 2. 좌/우 스테레오 이미지 불러오기
# -------------------------------------------------------
# 스테레오 카메라에서 촬영된 좌측 / 우측 이미지
left_color = cv2.imread("./images/left.png")
right_color = cv2.imread("./images/right.png")

# 이미지 로딩 실패 시 예외 처리
if left_color is None or right_color is None:
    raise FileNotFoundError("좌/우 이미지를 찾지 못했습니다.")

# -------------------------------------------------------
# 3. 카메라 파라미터 설정
# -------------------------------------------------------
# f : 카메라 초점거리 (pixel 단위)
# B : 두 카메라 사이 거리 (baseline, meter)
f = 700.0
B = 0.12

# -------------------------------------------------------
# 4. 관심 영역(ROI) 설정
# -------------------------------------------------------
# ROI = Region Of Interest
# 특정 물체 영역에서 평균 disparity / depth를 계산하기 위해 설정
# 형식: (x좌표, y좌표, width, height)
rois = {
    "Painting": (55, 50, 130, 110),
    "Frog": (90, 265, 230, 95),
    "Teddy": (310, 35, 115, 90)
}

# -------------------------------------------------------
# 5. 그레이스케일 변환
# -------------------------------------------------------
# StereoBM 알고리즘은 grayscale 이미지를 사용
left_gray = cv2.cvtColor(left_color, cv2.COLOR_BGR2GRAY)
right_gray = cv2.cvtColor(right_color, cv2.COLOR_BGR2GRAY)

# -------------------------------------------------------
# 6. Disparity 계산
# -------------------------------------------------------
# disparity = 좌/우 이미지 픽셀 위치 차이
# disparity가 클수록 물체가 카메라에 가까움

# disparity 탐색 범위 (16의 배수여야 함)
num_disparities = 16 * 6

# 블록 매칭 크기 (홀수)
block_size = 15

# Stereo Block Matching 알고리즘 생성
stereo = cv2.StereoBM_create(
    numDisparities=num_disparities,
    blockSize=block_size
)

# disparity 계산
# OpenCV는 결과를 16배 스케일된 값으로 반환하기 때문에 16으로 나눔
disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0

# -------------------------------------------------------
# 7. Depth 계산
# -------------------------------------------------------
# Depth 공식
# Z = fB / d
# f : focal length
# B : baseline
# d : disparity

depth_map = np.zeros_like(disparity, dtype=np.float32)

# disparity가 0 이하인 값은 무효
valid_mask = disparity > 0

# 유효한 픽셀만 depth 계산
depth_map[valid_mask] = (f * B) / disparity[valid_mask]

# -------------------------------------------------------
# 8. ROI별 평균 disparity / depth 계산
# -------------------------------------------------------
results = {}

for name, (x, y, w, h) in rois.items():

    # ROI 영역 추출
    roi_disp = disparity[y:y+h, x:x+w]
    roi_depth = depth_map[y:y+h, x:x+w]

    # disparity가 0보다 큰 유효 픽셀만 사용
    roi_valid = roi_disp > 0

    if np.any(roi_valid):
        mean_disp = np.mean(roi_disp[roi_valid])
        mean_depth = np.mean(roi_depth[roi_valid])
    else:
        mean_disp = 0.0
        mean_depth = 0.0

    # 결과 저장
    results[name] = {
        "mean_disparity": mean_disp,
        "mean_depth": mean_depth
    }

# -------------------------------------------------------
# 9. 결과 출력
# -------------------------------------------------------
print("=== ROI별 평균 Disparity / Depth ===")

for name, values in results.items():
    print(f"{name}")
    print(f"  Mean Disparity : {values['mean_disparity']:.3f}")
    print(f"  Mean Depth     : {values['mean_depth']:.3f} m")

# disparity가 존재하는 ROI만 필터링
valid_results = {k: v for k, v in results.items() if v["mean_disparity"] > 0}

# 가장 가까운 물체 / 가장 먼 물체 판단
if len(valid_results) > 0:
    nearest = max(valid_results.items(), key=lambda x: x[1]["mean_disparity"])
    farthest = max(valid_results.items(), key=lambda x: x[1]["mean_depth"])

    print("\n=== 해석 ===")
    print(f"가장 가까운 ROI: {nearest[0]}")
    print(f"가장 먼 ROI: {farthest[0]}")
else:
    print("\n유효한 disparity가 있는 ROI가 없습니다.")

# -------------------------------------------------------
# 10. disparity 시각화
# -------------------------------------------------------
# disparity 값들을 정규화하여 컬러맵으로 표현
disp_tmp = disparity.copy()
disp_tmp[disp_tmp <= 0] = np.nan

if np.all(np.isnan(disp_tmp)):
    raise ValueError("유효한 disparity 값이 없습니다.")

d_min = np.nanpercentile(disp_tmp, 5)
d_max = np.nanpercentile(disp_tmp, 95)

if d_max <= d_min:
    d_max = d_min + 1e-6

# disparity 정규화
disp_scaled = (disp_tmp - d_min) / (d_max - d_min)
disp_scaled = np.clip(disp_scaled, 0, 1)

disp_vis = np.zeros_like(disparity, dtype=np.uint8)
valid_disp = ~np.isnan(disp_tmp)
disp_vis[valid_disp] = (disp_scaled[valid_disp] * 255).astype(np.uint8)

# 컬러맵 적용 (JET)
disparity_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

# -------------------------------------------------------
# 11. depth 시각화
# -------------------------------------------------------
depth_vis = np.zeros_like(depth_map, dtype=np.uint8)

if np.any(valid_mask):
    depth_valid = depth_map[valid_mask]

    z_min = np.percentile(depth_valid, 5)
    z_max = np.percentile(depth_valid, 95)

    if z_max <= z_min:
        z_max = z_min + 1e-6

    depth_scaled = (depth_map - z_min) / (z_max - z_min)
    depth_scaled = np.clip(depth_scaled, 0, 1)

    # depth는 값이 클수록 멀기 때문에 반전
    depth_scaled = 1.0 - depth_scaled
    depth_vis[valid_mask] = (depth_scaled[valid_mask] * 255).astype(np.uint8)

depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

# -------------------------------------------------------
# 12. ROI 표시
# -------------------------------------------------------
left_vis = left_color.copy()
right_vis = right_color.copy()

for name, (x, y, w, h) in rois.items():

    # 좌측 이미지 ROI 표시
    cv2.rectangle(left_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(left_vis, name, (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 우측 이미지 ROI 표시
    cv2.rectangle(right_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(right_vis, name, (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# -------------------------------------------------------
# 13. 결과 이미지 저장
# -------------------------------------------------------
cv2.imwrite(str(output_dir / "left_with_roi.png"), left_vis)
cv2.imwrite(str(output_dir / "right_with_roi.png"), right_vis)
cv2.imwrite(str(output_dir / "disparity_map.png"), disparity_color)
cv2.imwrite(str(output_dir / "depth_map.png"), depth_color)

# -------------------------------------------------------
# 14. 결과 화면 출력
# -------------------------------------------------------
cv2.imshow("Left with ROI", left_vis)
cv2.imshow("Right with ROI", right_vis)
cv2.imshow("Disparity Map", disparity_color)
cv2.imshow("Depth Map", depth_color)

cv2.waitKey(0)
cv2.destroyAllWindows()

print("\n결과 이미지 저장 완료:")
print(output_dir / "left_with_roi.png")
print(output_dir / "right_with_roi.png")
print(output_dir / "disparity_map.png")
print(output_dir / "depth_map.png")