import cv2 as cv  #opencv import
import numpy as np #numpy improt

def main():                                         #main 함수 정의
    img_path = "soccer.jpg"                         #불러올 이미지 경로
    img = cv.imread(img_path)                       #cv.imread()함수로 이미지 읽어와 img 변수에 저장 

    if img is None:                                 #img가 제대로 불러와지지 않은 경우
        print("이미지를 불러올 수 없습니다.")        #오류 메세지 출력
        return                                      #함수 실행 중단 및 종료 

    #이미지 크기 변경
    scale = 0.5                                     # 이미지 크기 축소 (원본의 50%)
    img = cv.resize(img, None, fx=scale, fy=scale)  #cv.resize()를 사용하여 이미지 크기 축소 
                                                    # none : 새로운 크기를 직접 지정하지 않음
                                                    #fx: 가로비율/fy: 세로비율
    #컬러 변환
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)      # cv.cvtColor()를 사용해 컬러 이미지를 그레이 스케일로 변환
                                                    # COLOR_BGR2GRAY는 BGR 이미지를 흑백 이미지로 변환하는 옵션

    # hstack 위해 채널 맞추기
    gray_bgr = cv.cvtColor(gray, cv.COLOR_GRAY2BGR) # gray 이미지는 채널이 1개이기 때문에 다시 3채널(BGR)로 변환

    combined = np.hstack((img, gray_bgr))           # hstack(): 두 이미지를 가로 방향으로 연결
                                                    # 왼쪽 원본, 오른쪽 그레이

    cv.imshow("Original | Grayscale", combined)     # imshow()로 화면에 이미지를 출력
                                                    # 창 제목은 "Original | Grayscale"
    cv.waitKey(0)                                   # 키보드 입력이 들어올 때까지 프로그램을 대기 (0은 무한 대기)
    cv.destroyAllWindows()                          # 열려있는 모든 OpenCV 창을 닫음

if __name__ == "__main__":                          # 현재 파일이 직접 실행된 경우에만 아래 코드를 실행
    main()                                          # main() 함수를 호출