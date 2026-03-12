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

---

## 전체 코드 (01_gray.py)

        import cv2 as cv
        import numpy as np

        def main():
            img_path = "soccer.jpg"
            img = cv.imread(img_path)

            if img is None:
                print("이미지를 불러올 수 없습니다.")
                return

            scale = 0.5
            img = cv.resize(img, None, fx=scale, fy=scale)

            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            gray_bgr = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)

            combined = np.hstack((img, gray_bgr))

            cv.imshow("Original | Grayscale", combined)
            cv.waitKey(0)
            cv.destroyAllWindows()

        if __name__ == "__main__":
            main()



---
# 2. 마우스 입력을 이용한 붓질 기능

## 문제

마우스를 이용하여 이미지 위에 그림을 그리고,
키보드 입력을 이용해 붓 크기를 조절하는 기능을 구현한다.


## 요구사항

- 초기 붓 크기 : 5
- "+" 입력 → 붓 크기 증가
- "-" 입력 → 붓 크기 감소
- 붓 크기 범위 : 1 ~ 15
- 좌클릭 → 파란색
- 우클릭 → 빨간색
- 드래그로 연속 그리기
- q 키 → 프로그램 종료


## 주요 코드

- cv.setMouseCallback(window, callback) : 마우스 이벤트 처리
- cv.circle(img, (x,y), radius, color, -1) : 붓 효과 구현
- cv.waitKey() : 키보드 입력 처리
- clamp() : 붓 크기를 최소/최대 범위 내로 제한

## 전체 코드 (02_paint.py)

        import cv2 as cv
        import numpy as np

        BRUSH_MIN = 1
        BRUSH_MAX = 15

        state = {
            "img": None,
            "drawing": False,
            "button": None,
            "brush": 5
        }

        def clamp(v, lo, hi):
            return max(lo, min(hi, v))

        def mouse_cb(event, x, y, flags, param):
            img = state["img"]
            if img is None:
                return

            if event == cv.EVENT_LBUTTONDOWN:
                state["drawing"] = True
                state["button"] = cv.EVENT_LBUTTONDOWN
                cv.circle(img, (x, y), state["brush"], (255, 0, 0), -1)

            elif event == cv.EVENT_RBUTTONDOWN:
                state["drawing"] = True
                state["button"] = cv.EVENT_RBUTTONDOWN
                cv.circle(img, (x, y), state["brush"], (0, 0, 255), -1)

            elif event == cv.EVENT_MOUSEMOVE:
                if state["drawing"]:
                    if state["button"] == cv.EVENT_LBUTTONDOWN:
                        cv.circle(img, (x, y), state["brush"], (255, 0, 0), -1)
                    elif state["button"] == cv.EVENT_RBUTTONDOWN:
                        cv.circle(img, (x, y), state["brush"], (0, 0, 255), -1)

            elif event in (cv.EVENT_LBUTTONUP, cv.EVENT_RBUTTONUP):
                state["drawing"] = False
                state["button"] = None

        def main():
            img_path = "soccer.jpg"
            img = cv.imread(img_path)

            if img is None:
                print(f"[ERROR] 이미지를 불러오지 못했습니다: {img_path}")
                return

            state["img"] = img

            win = "Paint on Image (+/- brush, q quit)"
            cv.namedWindow(win)
            cv.setMouseCallback(win, mouse_cb)

            while True:
                view = state["img"].copy()

                cv.putText(
                    view,
                    f"Brush: {state['brush']} (+/-) q:quit",
                    (10, 30),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 0),
                    2
                )

                cv.imshow(win, view)

                key = cv.waitKey(1) & 0xFF

                if key == ord('q'):
                    break
                elif key in (ord('+'), ord('=')):
                    state["brush"] = clamp(state["brush"] + 1, BRUSH_MIN, BRUSH_MAX)
                elif key in (ord('-'), ord('_')):
                    state["brush"] = clamp(state["brush"] - 1, BRUSH_MIN, BRUSH_MAX)

            cv.destroyAllWindows()

        if __name__ == "__main__":
            main()

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