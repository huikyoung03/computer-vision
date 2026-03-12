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

# 3. 마우스로영역선택및ROI(관심영역) 추출

## 문제

이미지를 불러온 후 마우스를 이용하여 드래그 방식으로
관심 영역(ROI, Region of Interest)을 선택한다.

선택된 영역은 별도의 창에 출력되며,
키보드 입력을 통해 영역을 저장하거나 초기화할 수 있도록 구현한다.



## 주요 코드

- cv.setMouseCallback() : 마우스 이벤트 처리
- cv.rectangle() : 선택 영역 표시
- numpy slicing : ROI 영역 추출
- cv.imwrite() : ROI 이미지 저장

## 결과 



##전체 코드 (03_roi.py)

        import cv2 as cv
