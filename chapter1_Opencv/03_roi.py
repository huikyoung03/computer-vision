import cv2 as cv                            #opencv import
import numpy as np                          #numpy improt
import time                                 # time 모듈 (파일 저장 시 시간 이름 생성)

# 프로그램의 현재 상태를 저장하기 위한 딕셔너리
state = {                   
    "img": None,             # 원본 이미지
    "img_disp": None,        # 화면에 표시할 이미지 (리셋 등에 사용)
    "dragging": False,       # 마우스로 드래그 중인지 여부
    "p0": None,              # 드래그 시작점 좌표 (x0, y0)
    "p1": None,              # 드래그 끝점 좌표 (x1, y1)
    "roi": None              # 선택된 관심 영역(ROI)을 저장
}

# 두 점을 받아서 올바른 사각형 좌표로 정렬하는 함수
def norm_rect(p0, p1):      
    x0, y0 = p0              # 시작점 좌표를 x0, y0로 분리
    x1, y1 = p1              # 끝점 좌표를 x1, y1로 분리
    x_min, x_max = sorted([x0, x1])  # x좌표를 정렬하여 왼쪽과 오른쪽을 구분
    y_min, y_max = sorted([y0, y1])  # y좌표를 정렬하여 위와 아래를 구분
    return x_min, y_min, x_max, y_max # 정렬된 사각형 좌표 반환

# 마우스 이벤트가 발생할 때 실행되는 콜백 함수
def mouse_cb(event, x, y, flags, param):  
    if state["img"] is None:              # 이미지가 로드되지 않았으면
        return                            # 함수 실행을 종료
    
    # 마우스 왼쪽 버튼을 눌렀을 때
    if event == cv.EVENT_LBUTTONDOWN:     
        state["dragging"] = True          # 드래그 시작 상태로 변경
        state["p0"] = (x, y)              # 시작 좌표 저장
        state["p1"] = (x, y)              # 끝 좌표도 초기값으로 설정

    # 드래그 중 마우스를 움직일 때
    elif event == cv.EVENT_MOUSEMOVE and state["dragging"]:  
        state["p1"] = (x, y)              # 현재 마우스 위치를 끝 좌표로 계속 업데이트

    # 마우스 왼쪽 버튼을 놓았을 때
    elif event == cv.EVENT_LBUTTONUP and state["dragging"]:  
        state["dragging"] = False         # 드래그 상태 종료
        state["p1"] = (x, y)              # 최종 끝 좌표 저장

        x_min, y_min, x_max, y_max = norm_rect(state["p0"], state["p1"])
                                         # 시작점과 끝점을 정렬하여 정상적인 사각형 좌표 생성

        # 너무 작은 선택 방지 (실수 클릭 방지)
        if (x_max - x_min) < 5 or (y_max - y_min) < 5:  # ROI 크기가 너무 작으면
            state["roi"] = None          # ROI를 None으로 설정
            return                       # ROI 생성 없이 종료

        roi = state["img"][y_min:y_max, x_min:x_max].copy() # NumPy 슬라이싱으로 관심 영역을 잘라냄
        state["roi"] = roi               # 잘라낸 ROI를 state에 저장
        cv.imshow("ROI", roi)            # ROI 이미지를 별도의 창에 출력

# ROI 선택 상태를 초기화하는 함수
def reset():                             
    state["img_disp"] = state["img"].copy()  # 표시용 이미지를 원본 이미지로 복사
    state["dragging"] = False                # 드래그 상태 초기화
    state["p0"] = None                       # 시작 좌표 초기화
    state["p1"] = None                       # 끝 좌표 초기화
    state["roi"] = None                      # ROI 정보 초기화

# 프로그램의 메인 함수
def main():                                 
    img_path = "soccer.jpg"                  # 불러올 이미지 파일 경로
    img = cv.imread(img_path)                # OpenCV imread()로 이미지 로드
    if img is None:                          # 이미지 로드 실패 시
        print(f"[ERROR] 이미지를 불러오지 못했습니다: {img_path}")  # 오류 메시지 출력
        return                               # 프로그램 종료

    state["img"] = img                       # 원본 이미지를 state에 저장
    reset()                                  # 초기 상태 설정

    win = "Select ROI (drag) | r:reset | s:save | q:quit"  # 창 이름 설정
    cv.namedWindow(win)                      # OpenCV 창 생성
    cv.setMouseCallback(win, mouse_cb)       # 해당 창에 마우스 이벤트 콜백 함수 등록

    while True:                              # 무한 반복 루프
        frame = state["img"].copy()          # 현재 이미지를 복사하여 frame에 저장

        # 드래그 중이면 사각형을 표시
        if state["p0"] is not None and state["p1"] is not None:  # 시작점과 끝점이 존재하면
            x_min, y_min, x_max, y_max = norm_rect(state["p0"], state["p1"])
                                                             # 사각형 좌표 정렬
            # dragging 여부와 상관없이 현재 선택 박스는 계속 보이게
            cv.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                                                             # 녹색 사각형으로 ROI 영역 표시

        cv.putText(frame, "Drag to select ROI | r reset | s save | q quit",
                   (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                                                             # 화면 상단에 사용 안내 텍스트 표시

        cv.imshow(win, frame)                                # 현재 프레임을 화면에 출력

        key = cv.waitKey(1) & 0xFF                            # 키 입력을 1ms 동안 대기 후 key 값 저장
        if key == ord('q'):                                   # 'q' 키 입력 시
            break                                             # 반복문 종료 → 프로그램 종료
        elif key == ord('r'):                                 # 'r' 키 입력 시
            reset()                                           # ROI 선택 상태 초기화
            # ROI 창이 떠있으면 닫기
            try:
                cv.destroyWindow("ROI")                       # ROI 창이 존재하면 닫기
            except cv.error:                                  # ROI 창이 없는 경우 발생하는 예외 처리
                pass                                          # 아무 작업도 하지 않고 넘어감
        elif key == ord('s'):                                 # 's' 키 입력 시
            if state["roi"] is None:                          # ROI가 선택되지 않았다면
                print("[INFO] 저장할 ROI가 없습니다. 먼저 드래그로 영역을 선택하세요.")
                                                             # 안내 메시지 출력
            else:
                ts = time.strftime("%Y%m%d_%H%M%S")           # 현재 시간을 문자열로 생성
                out_path = f"roi_{ts}.png"                    # 저장할 파일 이름 생성
                ok = cv.imwrite(out_path, state["roi"])       # ROI 이미지를 파일로 저장
                print(f"[SAVE] {out_path} ({'OK' if ok else 'FAIL'})")
                                                             # 저장 결과 출력

    cv.destroyAllWindows()                                   # 모든 OpenCV 창을 닫음

if __name__ == "__main__":                                   # 현재 파일이 직접 실행된 경우
    main()                                                   # main() 함수 실행