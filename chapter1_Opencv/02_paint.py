import cv2 as cv                                #opencv import
import numpy as np                              #numpy improt

BRUSH_MIN = 1                                   # 붓 크기의 최소값을 1로 설정
BRUSH_MAX = 15                                  # 붓 크기의 최대값을 15로 설정

state = {                                       # 프로그램 상태를 저장하기 위한 딕셔너리
    "img": None,                                # 그림이 그려질 이미지
    "drawing": False,                           # 현재 마우스를 누른 상태에서 그리는 중인지 여부
    "button": None,                             # 어떤 마우스 버튼이 눌렸는지 저장 (좌클릭 또는 우클릭)
    "brush": 5                                  # 초기 붓 크기 설정
}

# 값을 특정 범위(lo~hi) 안으로 제한하는 함수
def clamp(v, lo, hi):        
    return max(lo, min(hi, v))# v가 hi보다 크면 hi로 제한하고, lo보다 작으면 lo로 제한

# 마우스 이벤트가 발생할 때 호출되는 콜백 함수
def mouse_cb(event, x, y, flags, param):  
    img = state["img"]       # state 딕셔너리에서 현재 이미지를 가져옴
    if img is None:          # 이미지가 없는 경우
        return               # 함수 실행을 종료

    # 좌클릭: 파란색, 우클릭: 빨간색 (OpenCV는 BGR)
    # 마우스 왼쪽 버튼을 눌렀을 때
    if event == cv.EVENT_LBUTTONDOWN:                           
        state["drawing"] = True                                 #그림 그리는 상태 변경
        state["button"] = cv.EVENT_LBUTTONDOWN                  # 눌린 버튼 정보 저장
        cv.circle(img, (x, y), state["brush"], (255, 0, 0), -1) # 현재 위치(x,y)에 파란색 원을 그림
                                                                # 현재 붓 크기, -1: 내부 채우기
    # 마우스 오른쪽 버튼을 눌렀을 때
    elif event == cv.EVENT_RBUTTONDOWN:                         
        state["drawing"] = True                                 # 그림을 그리는 상태 변경
        state["button"] = cv.EVENT_RBUTTONDOWN                  # 눌린 버튼 정보 저장
        cv.circle(img, (x, y), state["brush"], (0, 0, 255), -1) # 현재 위치(x,y)에 빨간색 원을 그림

    # 마우스를 움직일 때 발생하는 이벤트
    elif event == cv.EVENT_MOUSEMOVE:                           
        if state["drawing"]:                                    # 마우스 버튼이 눌린 상태라면
            if state["button"] == cv.EVENT_LBUTTONDOWN:         # 왼쪽 버튼으로 그리는 중이면
                cv.circle(img, (x, y), state["brush"], (255, 0, 0), -1) # 파란색 원을 계속 그림
            elif state["button"] == cv.EVENT_RBUTTONDOWN:       # 오른쪽 버튼으로 그리는 중이면
                cv.circle(img, (x, y), state["brush"], (0, 0, 255), -1) # 빨간색 원을 계속 그림

# 마우스 버튼을 뗐을 때
    elif event in (cv.EVENT_LBUTTONUP, cv.EVENT_RBUTTONUP):
        state["drawing"] = False          # 그림 그리는 상태 종료
        state["button"] = None            # 어떤 버튼도 눌려있지 않음

# 프로그램의 메인 함수
def main():                               
    img_path = "soccer.jpg"               # 사용할 이미지 파일 경로
    img = cv.imread(img_path)             # OpenCV imread()로 이미지를 읽어옴

    if img is None:                       # 이미지 로드 실패 시
        print(f"[ERROR] 이미지를 불러오지 못했습니다: {img_path}")  
                                          # 오류 메시지 출력
        return                            # 프로그램 종료

    state["img"] = img                    # state 딕셔너리에 이미지 저장

    win = "Paint on Image (+/- brush, q quit)" # 창의 제목
    cv.namedWindow(win)                   # 창 생성
    cv.setMouseCallback(win, mouse_cb)    # 마우스 이벤트가 발생하면 mouse_cb 함수 호출

    # 프로그램을 계속 실행하는 반복문
    while True:                           
        # 안내 텍스트만 오버레이해서 보여주기 (실제 그림은 img에 누적됨)
        view = state["img"].copy()        # 원본 이미지를 복사하여 view 변수에 저장

        cv.putText(
            view,                         # 텍스트를 표시할 이미지
            f"Brush: {state['brush']}  (+/-)  Left:Blue  Right:Red  q:quit",
                                          # 화면에 표시할 안내 문구
            (10, 30),                     # 텍스트가 시작될 위치 좌표
            cv.FONT_HERSHEY_SIMPLEX,      # 사용할 글꼴
            0.8,                          # 글자 크기
            (0, 0, 0),                    # 글자 색상 (검정색)
            2                             # 글자 두께
        )

        cv.imshow(win, view)              # 현재 이미지를 창에 출력

        key = cv.waitKey(1) & 0xFF        # 1ms 동안 키 입력을 기다리고 입력값을 key에 저장

        if key == ord('q'):               # 'q' 키가 눌리면
            break                         # 반복문 종료 → 프로그램 종료
        elif key == ord('+'):             # '+' 키가 눌리면
            state["brush"] = clamp(state["brush"] + 1, BRUSH_MIN, BRUSH_MAX)
                                          # 붓 크기를 1 증가시키고 최대값 15로 제한
        elif key == ord('-'):             # '-' 키가 눌리면
            state["brush"] = clamp(state["brush"] - 1, BRUSH_MIN, BRUSH_MAX)
                                          # 붓 크기를 1 감소시키고 최소값 1로 제한

    cv.destroyAllWindows()                # 열려 있는 모든 OpenCV 창을 닫음

if __name__ == "__main__":                # 현재 파일이 직접 실행된 경우에만 아래 코드를 실행
    main()                                # main() 함수를 호출
