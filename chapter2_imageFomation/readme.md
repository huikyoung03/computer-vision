# 1. 체크보드 기반 카메라 캘리브레이션

## 문제

이미지에서 체크보드 코너를 검출하고 실제 좌표와 이미지 좌표의 대응 관계를 이용하여 카메라의 내부 파라미터를 추정한다.
체크보드 패턴이 촬영된 여러 장의 이미지를 이용하여 카메라의 내부 행렬과 왜곡 계수를 계산하여 왜곡을 보정한다.

## 요구사항

- 모든 이미지에서 체크보드 코너를 검출한다.
- 체크보드의 실제좌표와 이미지에서 찾은 코너 좌표를 구성한다.
- cv2.calibrateCamera()를 사용하여 카메라 내부 행렬k와 왜곡 계수를 구한다.
- cv2.undistort()를 사용하여 왜곡 보정한 결과를 시각화한다.

## 전체 코드 (01_calibration.py)

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



## 주요 코드 


        

## 결과 화면 
<img width="755" height="316" alt="image" src="https://github.com/user-attachments/assets/bee6fdcf-0e65-437e-8d7d-79778294ac72" />

![calibration_result](https://github.com/user-attachments/assets/f31f57bc-664f-49b7-b808-330ff40e012f)

---
# 2. 이미지 Rotation & Transformation

## 문제

한 장의 이미지에 회전, 크기 조절, 평행 이동을 적용한다.


## 요구사항

- 이미지의 중심 기준으로 +30도 회전
- 회전과 동시에 크기를 0.8로 조절
- 그 결과를 x축 방향으로 +80px, y축 방향으로 -40px만큼 평행이동

## 전체 코드 (02_rotation_transformation.py)

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


## 주요 코드 
        
## 결과 화면 

![rotation_transformation_result](https://github.com/user-attachments/assets/a07b9c88-08bb-4689-9b0d-3c6075b9456c)

---

# 3. stereo disparity 기반 depth 추정 

## 문제

같은 장면을 왼쪽, 오른쪽 두 카메라에서 촬영한 두 장의 이미지를 활용해 깊이를 추정 
두 이미지에서 같은 물체가 얼마나 옆으로 이동해 보이는지 계산하여 물체가 카메라에서 얼마나 떨어져 있는지 (depth)를 구함

## 요구사항

- 입력 이미지를 그레이 스케일로 변환 후, cv2.StereoBM_create()를 사용하여 disparity map 계산
- Disparity >0 인 픽셀만 사용하여 depth map 계산
- ROI Painting, Frog, Teddy 각각에 대해 평균 disparity와 평균 depth를 계산
- 세 ROI중 어떤 영역이 가장 가까운지, 어떤 영역이 가장 먼지 해

## 전체 코드 (03_depth.py)
        import cv2
        import numpy as np
        from pathlib import Path
        
        # 출력 폴더 생성
        output_dir = Path("./outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 좌/우 이미지 불러오기
        left_color = cv2.imread("./images/left.png")
        right_color = cv2.imread("./images/right.png")
        
        if left_color is None or right_color is None:
            raise FileNotFoundError("좌/우 이미지를 찾지 못했습니다.")
        
        # 카메라 파라미터
        f = 700.0
        B = 0.12
        
        # ROI 설정
        rois = {
            "Painting": (55, 50, 130, 110),
            "Frog": (90, 265, 230, 95),
            "Teddy": (310, 35, 115, 90)
        }
        
        # 그레이스케일 변환
        left_gray = cv2.cvtColor(left_color, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_color, cv2.COLOR_BGR2GRAY)
        
        # -----------------------------
        # 1. Disparity 계산
        # -----------------------------
        num_disparities = 16 * 6   # 16의 배수
        block_size = 15            # 홀수
        
        stereo = cv2.StereoBM_create(
            numDisparities=num_disparities,
            blockSize=block_size
        )
        
        disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
        
        # -----------------------------
        # 2. Depth 계산
        # Z = fB / d
        # -----------------------------
        depth_map = np.zeros_like(disparity, dtype=np.float32)
        
        valid_mask = disparity > 0
        depth_map[valid_mask] = (f * B) / disparity[valid_mask]
        
        # -----------------------------
        # 3. ROI별 평균 disparity / depth 계산
        # -----------------------------
        results = {}
        
        for name, (x, y, w, h) in rois.items():
            roi_disp = disparity[y:y+h, x:x+w]
            roi_depth = depth_map[y:y+h, x:x+w]
        
            roi_valid = roi_disp > 0
        
            if np.any(roi_valid):
                mean_disp = np.mean(roi_disp[roi_valid])
                mean_depth = np.mean(roi_depth[roi_valid])
            else:
                mean_disp = 0.0
                mean_depth = 0.0
        
            results[name] = {
                "mean_disparity": mean_disp,
                "mean_depth": mean_depth
            }
        
        # -----------------------------
        # 4. 결과 출력
        # -----------------------------
        print("=== ROI별 평균 Disparity / Depth ===")
        for name, values in results.items():
            print(f"{name}")
            print(f"  Mean Disparity : {values['mean_disparity']:.3f}")
            print(f"  Mean Depth     : {values['mean_depth']:.3f} m")
        
        valid_results = {k: v for k, v in results.items() if v["mean_disparity"] > 0}
        
        if len(valid_results) > 0:
            nearest = max(valid_results.items(), key=lambda x: x[1]["mean_disparity"])
            farthest = max(valid_results.items(), key=lambda x: x[1]["mean_depth"])
        
            print("\n=== 해석 ===")
            print(f"가장 가까운 ROI: {nearest[0]}")
            print(f"가장 먼 ROI: {farthest[0]}")
        else:
            print("\n유효한 disparity가 있는 ROI가 없습니다.")
        
        # -----------------------------
        # 5. disparity 시각화
        # 가까울수록 빨강 / 멀수록 파랑
        # -----------------------------
        disp_tmp = disparity.copy()
        disp_tmp[disp_tmp <= 0] = np.nan
        
        if np.all(np.isnan(disp_tmp)):
            raise ValueError("유효한 disparity 값이 없습니다.")
        
        d_min = np.nanpercentile(disp_tmp, 5)
        d_max = np.nanpercentile(disp_tmp, 95)
        
        if d_max <= d_min:
            d_max = d_min + 1e-6
        
        disp_scaled = (disp_tmp - d_min) / (d_max - d_min)
        disp_scaled = np.clip(disp_scaled, 0, 1)
        
        disp_vis = np.zeros_like(disparity, dtype=np.uint8)
        valid_disp = ~np.isnan(disp_tmp)
        disp_vis[valid_disp] = (disp_scaled[valid_disp] * 255).astype(np.uint8)
        
        disparity_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
        
        # -----------------------------
        # 6. depth 시각화
        # 가까울수록 빨강 / 멀수록 파랑
        # -----------------------------
        depth_vis = np.zeros_like(depth_map, dtype=np.uint8)
        
        if np.any(valid_mask):
            depth_valid = depth_map[valid_mask]
        
            z_min = np.percentile(depth_valid, 5)
            z_max = np.percentile(depth_valid, 95)
        
            if z_max <= z_min:
                z_max = z_min + 1e-6
        
            depth_scaled = (depth_map - z_min) / (z_max - z_min)
            depth_scaled = np.clip(depth_scaled, 0, 1)
        
            # depth는 클수록 멀기 때문에 반전
            depth_scaled = 1.0 - depth_scaled
            depth_vis[valid_mask] = (depth_scaled[valid_mask] * 255).astype(np.uint8)
        
        depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        
        # -----------------------------
        # 7. Left / Right 이미지에 ROI 표시
        # -----------------------------
        left_vis = left_color.copy()
        right_vis = right_color.copy()
        
        for name, (x, y, w, h) in rois.items():
            cv2.rectangle(left_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(left_vis, name, (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
            cv2.rectangle(right_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(right_vis, name, (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # -----------------------------
        # 8. 저장
        # -----------------------------
        cv2.imwrite(str(output_dir / "left_with_roi.png"), left_vis)
        cv2.imwrite(str(output_dir / "right_with_roi.png"), right_vis)
        cv2.imwrite(str(output_dir / "disparity_map.png"), disparity_color)
        cv2.imwrite(str(output_dir / "depth_map.png"), depth_color)
        
        # -----------------------------
        # 9. 출력
        # -----------------------------
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
        ## 주요 코드
        
        - cv.setMouseCallback() : 마우스 이벤트 처리
        - cv.rectangle() : 선택 영역 표시
        - numpy slicing : ROI 영역 추출
        - cv.imwrite() : ROI 이미지 저장

## 결과 

<img width="1369" height="1221" alt="image" src="https://github.com/user-attachments/assets/9eab061f-c73b-4d35-a755-f2a9535c1607" />

<img width="628" height="367" alt="image" src="https://github.com/user-attachments/assets/575c785c-ed49-47cc-92fa-697eb05eb6a7" />


