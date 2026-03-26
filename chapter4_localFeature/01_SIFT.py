import cv2 as cv
import matplotlib.pyplot as plt

# 1. 이미지 로드 (mot_color70.jpg)
img = cv.imread('mot_color70.jpg')  
# → 이미지를 BGR 컬러 형식으로 읽어옴 (OpenCV 기본 포맷)
# → 실패 시 None 반환 (경로 확인 필요)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  
# → 컬러 이미지를 그레이스케일(단일 채널)로 변환
# → SIFT는 intensity 기반이므로 grayscale 입력 사용

# 2. SIFT 객체 생성 (특징점 최대 개수 제한)
sift = cv.SIFT_create(nfeatures=3000)  
# → SIFT(Scale-Invariant Feature Transform) 알고리즘 생성
# → nfeatures=3000 : 검출할 특징점(keypoint)의 최대 개수 제한
# → scale / rotation 변화에 강인한 특징점 검출 알고리즘

# 3. 특징점 검출 + 디스크립터 생성
keypoints, descriptors = sift.detectAndCompute(gray, None)  
# → detect: 특징점 위치(keypoint) 찾기
# → compute: 각 특징점에 대한 descriptor(128차원 벡터) 생성
# → keypoints: 위치(x, y), 크기(scale), 방향(angle) 정보 포함 객체 리스트
# → descriptors: (N x 128) numpy 배열 (각 keypoint의 특징 벡터)

# 4. 특징점 시각화 (크기 + 방향까지 표시)
img_with_kp = cv.drawKeypoints(
    img,                  # 원본 이미지
    keypoints,            # 검출된 특징점 리스트
    None,                 # 출력 이미지 (None이면 새로 생성)
    flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)
# → DRAW_RICH_KEYPOINTS 옵션:
#    - 원으로 keypoint의 scale(크기) 표시
#    - 방향(angle)을 선으로 표시
# → 단순 점이 아니라 특징의 구조까지 시각화됨

# 5. 결과 출력 (matplotlib 사용)
plt.figure(figsize=(12, 6))  
# → 전체 figure 크기 설정

plt.subplot(1, 2, 1)  
# → 1행 2열 중 첫 번째 subplot

plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))  
# → OpenCV(BGR) → matplotlib(RGB)로 색상 변환
plt.title('Original Image')  
plt.axis('off')  
# → 축 제거 (이미지 시각화용)

plt.subplot(1, 2, 2)  
# → 두 번째 subplot

plt.imshow(cv.cvtColor(img_with_kp, cv.COLOR_BGR2RGB))  
# → 특징점이 표시된 이미지 출력
plt.title('SIFT Keypoints (Rich)')  
plt.axis('off')

plt.show()  
# → 화면에 결과 출력