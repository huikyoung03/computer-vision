# 1. 이미지 불러오기 및 그레이스케일 변환

## 문제

OpenCV를 사용하여 이미지를 불러오고, 그레이스케일로 변환한 후
원본 이미지와 그레이스케일 이미지를 가로로 연결하여 출력한다.

## 주요 코드

        cv.imread(path)
->이미지 파일을 읽어 BGR 형식으로 로드

        cv.cvtColor(img, cv.COLOR_BGR2GRAY)
-> 컬러(BGR) → 그레이스케일(1채널) 변환

        np.hstack([img1, img2]) 
-> 두 이미지를 가로로 붙임 (단, 두 이미지의 높이/채널 수가 동일해야 함)

## 실행 결과 

<img width="2162" height="765" alt="image" src="https://github.com/user-attachments/assets/4e767731-9d96-4cea-ac7b-616f93feaf71" />


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

        cv.setMouseCallback(window, callback)
→ OpenCV 창에서 발생하는 마우스 이벤트를 처리하는 함수

        cv.circle(img, center, radius, color, thickness)
→ 마우스 위치에 원을 그려 붓 효과를 구현

        clamp()
-> 붓 크기를 최소/최대 범위 내로 제한

        cv.waitKey()
→ 키보드 입력을 받아 붓 크기 조절 및 프로그램 종료 처리

        cv.imshow()
→ 현재 이미지를 화면에 출력


## 실행 결과 

<img width="2148" height="1464" alt="image" src="https://github.com/user-attachments/assets/d84bf358-d55a-4226-ab0c-ca9b92956eda" />


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
                elif key == ord('+'):
                    state["brush"] = clamp(state["brush"] + 1, BRUSH_MIN, BRUSH_MAX)
                elif key == ord('-'):
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

        cv.setMouseCallback(window, callback)
→ 마우스 드래그 이벤트를 처리하여 ROI 선택

        cv.rectangle(img, pt1, pt2, color, thickness)
→ 선택한 영역을 사각형으로 화면에 표시

        numpy slicing
→ 이미지 배열에서 ROI 영역을 추출

        roi = img[y_min:y_max, x_min:x_max]

        cv.imwrite(filename, img)
→ 선택한 ROI 이미지를 파일로 저장


## 실행 결과 

<img width="2128" height="1472" alt="image" src="https://github.com/user-attachments/assets/f32fbe21-31b1-4d9d-bd4d-e266f061f368" />


## 전체 코드 (03_roi.py)

        import cv2 as cv
        import numpy as np
        import time

        state = {
            "img": None,
            "img_disp": None,
            "dragging": False,
            "p0": None,
            "p1": None,
            "roi": None
        }

        def norm_rect(p0, p1):
            x0, y0 = p0
            x1, y1 = p1
            x_min, x_max = sorted([x0, x1])
            y_min, y_max = sorted([y0, y1])
            return x_min, y_min, x_max, y_max

        def mouse_cb(event, x, y, flags, param):
            if state["img"] is None:
                return

            if event == cv.EVENT_LBUTTONDOWN:
                state["dragging"] = True
                state["p0"] = (x, y)
                state["p1"] = (x, y)

            elif event == cv.EVENT_MOUSEMOVE and state["dragging"]:
                state["p1"] = (x, y)

            elif event == cv.EVENT_LBUTTONUP and state["dragging"]:
                state["dragging"] = False
                state["p1"] = (x, y)

                x_min, y_min, x_max, y_max = norm_rect(state["p0"], state["p1"])

                if (x_max - x_min) < 5 or (y_max - y_min) < 5:
                    state["roi"] = None
                    return

                roi = state["img"][y_min:y_max, x_min:x_max].copy()
                state["roi"] = roi
                cv.imshow("ROI", roi)

        def reset():
            state["img_disp"] = state["img"].copy()
            state["dragging"] = False
            state["p0"] = None
            state["p1"] = None
            state["roi"] = None

        def main():
            img_path = "soccer.jpg"
            img = cv.imread(img_path)

            if img is None:
                print(f"[ERROR] 이미지를 불러오지 못했습니다: {img_path}")
                return

            state["img"] = img
            reset()

            win = "Select ROI (drag) | r:reset | s:save | q:quit"
            cv.namedWindow(win)
            cv.setMouseCallback(win, mouse_cb)

            while True:
                frame = state["img"].copy()

                if state["p0"] is not None and state["p1"] is not None:
                    x_min, y_min, x_max, y_max = norm_rect(state["p0"], state["p1"])
                    cv.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                cv.putText(frame, "Drag to select ROI | r reset | s save | q quit",
                        (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

                cv.imshow(win, frame)

                key = cv.waitKey(1) & 0xFF

                if key == ord('q'):
                    break
                elif key == ord('r'):
                    reset()
                    try:
                        cv.destroyWindow("ROI")
                    except cv.error:
                        pass
                elif key == ord('s'):
                    if state["roi"] is None:
                        print("[INFO] 저장할 ROI가 없습니다.")
                    else:
                        ts = time.strftime("%Y%m%d_%H%M%S")
                        out_path = f"roi_{ts}.png"
                        cv.imwrite(out_path, state["roi"])

            cv.destroyAllWindows()

        if __name__ == "__main__":
            main()
