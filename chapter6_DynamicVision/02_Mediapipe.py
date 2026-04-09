import cv2  # OpenCV: 영상 처리 및 카메라 입력
import mediapipe as mp  # Mediapipe: 얼굴 landmark 추출 라이브러리

# Mediapipe의 face mesh 모듈과 drawing 유틸 가져오기
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# FaceMesh 객체 생성
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,  # False: 영상(실시간) 모드, True: 이미지 한 장 처리
    max_num_faces=1,          # 최대 인식할 얼굴 개수
    refine_landmarks=True     # True: 눈, 입 주변 정밀 landmark 추가
)

# 웹캠 열기 (0 = 기본 카메라)
cap = cv2.VideoCapture(0)

while True:
    # 한 프레임 읽기
    ret, frame = cap.read()

    # 프레임을 못 읽으면 종료
    if not ret:
        break

    # OpenCV는 BGR, Mediapipe는 RGB 사용 → 변환 필요
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 얼굴 landmark 추출
    result = face_mesh.process(rgb)

    # 얼굴이 검출되었으면
    if result.multi_face_landmarks:
        # 여러 얼굴 중 하나씩 처리
        for face_landmarks in result.multi_face_landmarks:
            
            # 얼굴 landmark를 화면에 그리기
            mp_drawing.draw_landmarks(
                frame,                       # 출력할 이미지
                face_landmarks,              # landmark 정보 (468개 점)
                mp_face_mesh.FACEMESH_CONTOURS  # 얼굴 윤곽선 연결 방식
            )

    # 결과 화면 출력
    cv2.imshow("FaceMesh", frame)

    # 키 입력 대기 (1ms)
    # ESC 키(27) 누르면 종료
    if cv2.waitKey(1) == 27:
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()