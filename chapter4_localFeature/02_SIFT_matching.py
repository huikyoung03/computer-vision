import cv2 as cv
import matplotlib.pyplot as plt

# 1. 두 개의 이미지 불러오기
img1 = cv.imread('mot_color70.jpg')  
# → 첫 번째 이미지 로드 (BGR 형식)
# → 특징점 기준이 되는 기준 이미지

img2 = cv.imread('mot_color83.jpg')  
# → 두 번째 이미지 로드
# → 비교 대상 이미지 (다른 시점/각도일 가능성 있음)


# 2. SIFT 특징점 추출
sift = cv.SIFT_create()  
# → SIFT 객체 생성 (scale, rotation invariant 특징 추출)

kp1, des1 = sift.detectAndCompute(img1, None)  
# → img1에서 특징점 검출 + descriptor 생성
# → kp1: keypoint 리스트 (위치, 크기, 방향 포함)
# → des1: 각 keypoint의 128차원 특징 벡터

kp2, des2 = sift.detectAndCompute(img2, None)  
# → img2에서도 동일하게 특징점 + descriptor 생성


# 3. BFMatcher 생성 및 매칭
bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)  
# → Brute-Force Matcher 생성
# → NORM_L2: SIFT descriptor는 float 기반 → 유클리드 거리 사용
# → crossCheck=True:
#    - A → B 매칭 + B → A 매칭이 서로 일치할 때만 인정
#    - 더 정확하지만 매칭 수는 줄어듦 (정밀도 ↑, recall ↓)

matches = bf.match(des1, des2)  
# → 모든 descriptor 쌍을 비교해서 가장 가까운 매칭 찾음
# → 결과: DMatch 객체 리스트
#    - queryIdx: img1 descriptor index
#    - trainIdx: img2 descriptor index
#    - distance: 두 descriptor 간 거리 (작을수록 유사)


# 4. 거리 기준으로 정렬 (좋은 매칭 우선)
matches = sorted(matches, key=lambda x: x.distance)  
# → distance가 작은 순서대로 정렬
# → 가장 유사한 특징점 매칭이 앞쪽에 위치


# 5. 매칭 결과 시각화
res = cv.drawMatches(
    img1, kp1,           # 첫 번째 이미지 + keypoints
    img2, kp2,           # 두 번째 이미지 + keypoints
    matches[:50],        # 상위 50개 매칭만 시각화
    None,
    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)
# → 두 이미지를 좌우로 붙이고 매칭된 점을 선으로 연결
# → NOT_DRAW_SINGLE_POINTS:
#    - 매칭되지 않은 keypoint는 표시하지 않음
# → 결과:
#    - 선 = 대응되는 특징점 쌍 (correspondence)


# 6. matplotlib으로 결과 출력
plt.figure(figsize=(15, 8))  
# → 출력 이미지 크기 설정

plt.imshow(cv.cvtColor(res, cv.COLOR_BGR2RGB))  
# → OpenCV(BGR) → matplotlib(RGB) 변환

plt.title('SIFT Feature Matching')  
plt.axis('off')  

plt.show()  
# → 최종 결과 출력