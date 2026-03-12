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



## 주요 코드 

    np.mgrid() 
-> 체크보드의 실제 좌표 격자 생성

    cv2.findChessboardCorners(image, corner 개수) 
-> 체크보드 코너 위치 검출 

    cv2.cornerSubPix()
코너 위치 정밀화

    cv2.calibrateCamera(실제 세계 좌표, 이미지 좌표, 이미지 크기, none, none) 
-> 카메라 내부 파라미터 및 왜곡 계수 계산

    cv2.undistort()
-> 렌즈 왜곡 보정

        

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

## 주요 코드 

    cv2.getRotationMatrix2D() 
-> 회전 + 스케일 변환 행렬 생성

    cv2.warpAffine()
-> affine 변환 적용


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


## 주요 코드 

    cv2.StereoBM_create() 
-> Stereo Block Matching 객체 생성
    
    stereo.compute() 
-> disparity map 계산
 
    depth_map = fB / d 
-> disparity를 depth로 변환
    
    - Depth 공식
    
        - Z = fB / d
        - f : focal length
        - B : baseline
        - d : disparity
        
    np.mean()
-> ROI 평균 disparity / depth 계산
    
    cv2.applyColorMap()
-> disparity/depth 시각화
    
    cv2.rectangle(), cv2.putText()
-> ROI 표시

## 결과 

<img width="1369" height="1221" alt="image" src="https://github.com/user-attachments/assets/9eab061f-c73b-4d35-a755-f2a9535c1607" />

<img width="628" height="367" alt="image" src="https://github.com/user-attachments/assets/575c785c-ed49-47cc-92fa-697eb05eb6a7" />


