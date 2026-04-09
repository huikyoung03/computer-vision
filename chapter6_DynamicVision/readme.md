# 1. SORT 알고리즘을활용한다중객체추적기구현

## 문제

YOLOv3를 이용하여 영상에서 객체를 검출하고, SORT 알고리즘을 적용하여 프레임 간 동일 객체를 추적한다. 각 객체에 고유 ID를 부여하여 지속적으로 추적 결과를 시각화한다.

## 요구사항

• YOLOv3를 이용한 객체 검출 수행

• OpenCV DNN 모듈 사용

• SORT 알고리즘 구현 (Kalman Filter + Hungarian Algorithm)

• 프레임 간 객체 매칭 및 ID 유지

• 결과를 영상으로 시각화

## 전체 코드 (01_SORT_tracking.py)
```python
import cv2  # OpenCV: 영상 읽기, 객체 검출, 화면 출력 등에 사용
import numpy as np  # 수치 연산 및 배열 처리
from scipy.optimize import linear_sum_assignment  # Hungarian Algorithm 구현 함수
from filterpy.kalman import KalmanFilter  # Kalman Filter 클래스

# =========================
# 설정값
# =========================
VIDEO_PATH = "slow_traffic_small.mp4"  # 입력 비디오 파일 경로
CFG_PATH = "yolov3.cfg"                # YOLOv3 네트워크 구조 파일
WEIGHTS_PATH = "yolov3.weights"        # YOLOv3 학습 가중치 파일
NAMES_PATH = "coco.names"              # COCO 데이터셋 클래스 이름 파일

CONF_THRESHOLD = 0.5   # 객체라고 인정할 최소 confidence 값
NMS_THRESHOLD = 0.4    # NMS(중복 박스 제거)에 사용할 threshold
IOU_THRESHOLD = 0.3    # detection과 tracker를 같은 객체로 볼 최소 IoU
MAX_AGE = 10           # detection이 안 잡혀도 tracker를 유지할 최대 프레임 수
MIN_HITS = 3           # tracker가 안정적으로 검출되었다고 보기 위한 최소 hit 수

# 필요하면 추적할 클래스만 지정
# COCO 전체 클래스 중 아래 클래스만 추적 대상으로 사용
TARGET_CLASSES = {"car", "bus", "truck", "motorbike", "person", "bicycle"}


# =========================
# IOU 계산
# =========================
def iou(bb_test, bb_gt):
    """
    두 bounding box 사이의 IoU(Intersection over Union)를 계산한다.

    bb 형식: [x1, y1, x2, y2]
    x1, y1 : 박스의 좌상단 좌표
    x2, y2 : 박스의 우하단 좌표

    IoU = 교집합 넓이 / 합집합 넓이
    값이 1에 가까울수록 두 박스가 많이 겹친다는 의미
    """
    # 두 박스가 겹치는 영역의 좌상단/우하단 좌표 계산
    xx1 = max(bb_test[0], bb_gt[0])
    yy1 = max(bb_test[1], bb_gt[1])
    xx2 = min(bb_test[2], bb_gt[2])
    yy2 = min(bb_test[3], bb_gt[3])

    # 겹치는 영역의 너비/높이 계산
    # 겹치지 않으면 0이 되도록 max(0.0, ...) 사용
    w = max(0.0, xx2 - xx1)
    h = max(0.0, yy2 - yy1)
    inter = w * h  # 교집합 넓이

    # 각 bounding box의 넓이 계산
    area1 = max(0.0, bb_test[2] - bb_test[0]) * max(0.0, bb_test[3] - bb_test[1])
    area2 = max(0.0, bb_gt[2] - bb_gt[0]) * max(0.0, bb_gt[3] - bb_gt[1])

    # 합집합 = area1 + area2 - 교집합
    union = area1 + area2 - inter

    # 분모가 0이 되는 예외 방지
    if union <= 0:
        return 0.0

    return inter / union


# =========================
# bbox 변환 함수
# SORT에서 자주 쓰는 방식
# [x1,y1,x2,y2] <-> [cx,cy,s,r]
# cx, cy: 중심점
# s: 면적
# r: 종횡비
# =========================
def convert_bbox_to_z(bbox):
    """
    일반 bounding box 형식 [x1, y1, x2, y2]를
    Kalman Filter 측정값 형식 [cx, cy, s, r]로 변환한다.

    cx, cy : 중심 좌표
    s      : 면적(scale)
    r      : 종횡비(aspect ratio = width / height)
    """
    w = bbox[2] - bbox[0]  # 박스 너비
    h = bbox[3] - bbox[1]  # 박스 높이
    x = bbox[0] + w / 2.0  # 중심 x 좌표
    y = bbox[1] + h / 2.0  # 중심 y 좌표
    s = w * h              # 박스 면적
    r = w / float(h + 1e-6)  # 종횡비, 0 나눗셈 방지 위해 작은 값 추가
    return np.array([x, y, s, r]).reshape((4, 1))  # column vector 형태로 반환


def convert_x_to_bbox(x):
    """
    Kalman Filter 상태값 [cx, cy, s, r, ...]를
    다시 일반 bounding box [x1, y1, x2, y2]로 변환한다.

    x는 Kalman 상태벡터이며 최소 앞 4개 값이 [cx, cy, s, r]라고 가정한다.
    """
    s = x[2]  # 면적
    r = x[3]  # 종횡비

    # s = w*h, r = w/h 이므로
    # w = sqrt(s*r), h = s/w 로 복원
    w = np.sqrt(max(s * r, 0))
    h = s / (w + 1e-6)

    # 중심 좌표와 너비/높이로 좌상단/우하단 복원
    x1 = x[0] - w / 2.0
    y1 = x[1] - h / 2.0
    x2 = x[0] + w / 2.0
    y2 = x[1] + h / 2.0

    return np.array([x1, y1, x2, y2]).reshape((1, 4))


# =========================
# KalmanBoxTracker
# =========================
class KalmanBoxTracker:
    """
    개별 객체 하나를 추적하는 클래스

    각 tracker는
    - 고유 ID를 가짐
    - Kalman Filter로 다음 위치를 예측함
    - 새 detection이 들어오면 상태를 보정함
    """
    count = 0  # tracker id를 순차적으로 부여하기 위한 클래스 변수

    def __init__(self, bbox, class_name="object"):
        """
        새 tracker 생성

        bbox: [x1, y1, x2, y2]
        class_name: 객체 클래스 이름 (예: car, person)
        """
        # 상태 차원 7, 측정 차원 4인 Kalman Filter 생성
        # 상태: [x, y, s, r, x_dot, y_dot, s_dot]
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        # 상태 전이 행렬 F
        # 현재 상태로부터 다음 상태를 예측하는 데 사용
        # x, y, s는 속도 성분 x_dot, y_dot, s_dot의 영향을 받음
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ], dtype=float)

        # 관측 행렬 H
        # 실제 detection으로 관측 가능한 값은 [x, y, s, r] 뿐임
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ], dtype=float)

        # 측정 노이즈 공분산 R 조정
        # s, r 쪽 측정값에 조금 더 불확실성을 둠
        self.kf.R[2:, 2:] *= 10.0

        # 상태 공분산 P 조정
        # 속도 항목 초기 불확실성을 크게 둠
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0

        # 프로세스 노이즈 공분산 Q 조정
        # 예측 과정에서의 잡음 정도 설정
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        # 초기 상태값의 앞 4개를 detection bbox로 초기화
        self.kf.x[:4] = convert_bbox_to_z(bbox)

        # 마지막으로 update된 후 지난 프레임 수
        self.time_since_update = 0

        # tracker에 고유 id 부여
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

        # 누적 detection 횟수
        self.hits = 0

        # 연속적으로 detection된 횟수
        self.hit_streak = 0

        # tracker가 생성된 이후 총 지난 프레임 수
        self.age = 0

        # 이 tracker가 추적하는 객체 클래스 이름
        self.class_name = class_name

    def update(self, bbox, class_name=None):
        """
        새로운 detection 결과를 이용해 tracker 상태를 보정한다.

        bbox: 새로 검출된 bounding box
        class_name: 새 클래스 이름(필요 시 갱신)
        """
        self.time_since_update = 0   # 방금 업데이트됐으므로 0으로 초기화
        self.hits += 1               # 총 hit 수 증가
        self.hit_streak += 1         # 연속 hit 수 증가

        # detection 값을 이용해 Kalman Filter 보정
        self.kf.update(convert_bbox_to_z(bbox))

        # 클래스명이 들어오면 갱신
        if class_name is not None:
            self.class_name = class_name

    def predict(self):
        """
        Kalman Filter를 사용해 다음 프레임에서의 객체 상태를 예측한다.
        detection이 없는 프레임에서도 tracker 위치를 추정하는 데 사용된다.
        """
        # 면적(s) + 면적 변화량(s_dot)이 0 이하가 되면 비정상적이므로 보정
        if (self.kf.x[2] + self.kf.x[6]) <= 0:
            self.kf.x[6] = 0

        # 예측 단계 수행
        self.kf.predict()

        # tracker 나이 증가
        self.age += 1

        # 이전 프레임에서 update가 없었다면 연속 hit streak 끊김
        if self.time_since_update > 0:
            self.hit_streak = 0

        # update 없이 지난 프레임 수 증가
        self.time_since_update += 1

        # 예측 결과를 bbox 형식으로 반환
        return convert_x_to_bbox(self.kf.x)[0]

    def get_state(self):
        """
        현재 tracker 상태를 bounding box 형식으로 반환한다.
        """
        return convert_x_to_bbox(self.kf.x)[0]


# =========================
# Hungarian matching
# =========================
def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    detection들과 tracker들을 IoU 기반으로 매칭한다.

    detections: [[x1,y1,x2,y2], ...]
    trackers:   [[x1,y1,x2,y2], ...]

    반환값:
    - matches: 매칭된 detection-tracker 인덱스 쌍
    - unmatched_detections: 매칭되지 않은 detection 인덱스들
    - unmatched_trackers: 매칭되지 않은 tracker 인덱스들
    """
    # tracker가 하나도 없으면 모든 detection은 unmatched
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0,), dtype=int)

    # detection 수 x tracker 수 크기의 IoU 행렬 생성
    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

    # 각 detection-tracker 쌍에 대해 IoU 계산
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk)

    # Hungarian Algorithm은 비용(cost)을 최소화하는 방식이므로
    # IoU가 클수록 좋은 매칭을 만들기 위해 cost = 1 - IoU 사용
    matched_indices = np.array(linear_sum_assignment(1 - iou_matrix)).T

    # 매칭되지 않은 detection 찾기
    unmatched_detections = []
    for d in range(len(detections)):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)

    # 매칭되지 않은 tracker 찾기
    unmatched_trackers = []
    for t in range(len(trackers)):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    # IoU threshold를 만족하는 실제 매칭만 남김
    matches = []
    for m in matched_indices:
        # IoU가 너무 낮으면 같은 객체가 아니라고 판단
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    # matches를 최종 numpy 배열로 정리
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


# =========================
# SORT 클래스
# =========================
class Sort:
    """
    전체 다중 객체 추적(Multi-Object Tracking)을 담당하는 클래스

    역할:
    1. 기존 tracker들의 위치 예측
    2. detection과 tracker를 Hungarian Algorithm으로 매칭
    3. 매칭된 tracker는 update
    4. 매칭되지 않은 detection은 새 tracker 생성
    5. 너무 오래 갱신되지 않은 tracker는 제거
    """
    def __init__(self, max_age=10, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age              # tracker 유지 최대 프레임 수
        self.min_hits = min_hits            # 안정적인 tracker로 인정할 최소 hit 수
        self.iou_threshold = iou_threshold  # 매칭에 사용할 최소 IoU
        self.trackers = []                  # 현재 활성 tracker 리스트
        self.frame_count = 0                # 현재까지 처리한 프레임 수

    def update(self, detections, class_names):
        """
        한 프레임에서 detection 결과를 받아 tracker 상태를 갱신한다.

        detections: [[x1,y1,x2,y2], ...]
        class_names: detections와 같은 순서의 클래스 이름 리스트
        """
        self.frame_count += 1  # 프레임 카운트 증가

        trks = []   # 예측된 tracker bbox 저장용
        to_del = [] # 삭제할 tracker 인덱스 저장용

        # 현재 모든 tracker에 대해 다음 위치 예측
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()

            # 예측 결과가 NaN이면 비정상 tracker로 판단하여 삭제 대상에 추가
            if np.any(np.isnan(pos)):
                to_del.append(t)
            else:
                trks.append(pos)

        trks = np.array(trks)

        # NaN tracker 제거
        for t in reversed(to_del):
            self.trackers.pop(t)

        # detection과 tracker를 IoU 기반 Hungarian matching으로 연결
        matches, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            detections, trks, self.iou_threshold
        )

        # 매칭된 tracker는 detection 결과로 update
        for m in matches:
            det_idx, trk_idx = m[0], m[1]
            self.trackers[trk_idx].update(detections[det_idx], class_names[det_idx])

        # 매칭되지 않은 detection은 새 tracker 생성
        for i in unmatched_dets:
            trk = KalmanBoxTracker(detections[i], class_names[i])
            self.trackers.append(trk)

        ret = []             # 최종 출력할 tracking 결과
        alive_trackers = []  # 아직 유지할 tracker들

        for trk in self.trackers:
            # 너무 오래 update되지 않은 tracker는 제거 대상
            if trk.time_since_update <= self.max_age:
                alive_trackers.append(trk)

            # 출력 조건:
            # 1) 현재 프레임에서 update된 tracker이어야 하고
            # 2) 충분히 안정적으로 검출되었거나
            #    아직 초기 프레임 구간이면 출력
            if (trk.time_since_update < 1) and (
                trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            ):
                d = trk.get_state()
                ret.append([int(d[0]), int(d[1]), int(d[2]), int(d[3]), trk.id, trk.class_name])

        # 살아남은 tracker만 유지
        self.trackers = alive_trackers

        # [x1, y1, x2, y2, track_id, class_name] 형식 리스트 반환
        return ret


# =========================
# YOLOv3 객체 검출
# =========================
def load_yolo():
    """
    YOLOv3 모델을 로드하고 출력 레이어 이름을 반환한다.
    """
    net = cv2.dnn.readNet(WEIGHTS_PATH, CFG_PATH)  # 가중치와 cfg 파일로 네트워크 생성
    layer_names = net.getLayerNames()              # 전체 레이어 이름 목록
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]  # 출력 레이어만 추출
    return net, output_layers


def load_classes():
    """
    coco.names 파일에서 클래스 이름들을 읽어 리스트로 반환한다.
    """
    with open(NAMES_PATH, "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f.readlines()]
    return classes


def detect_objects(frame, net, output_layers, classes):
    """
    한 프레임에서 YOLOv3를 이용해 객체를 검출한다.

    반환값:
    - final_detections: [[x1,y1,x2,y2], ...]
    - final_classes:    각 detection의 클래스 이름 리스트
    """
    height, width = frame.shape[:2]  # 프레임 높이, 너비

    # 프레임을 YOLO 입력용 blob으로 변환
    # 1/255.0: 픽셀 정규화
    # (416,416): YOLOv3 입력 크기
    # swapRB=True: OpenCV BGR -> RGB 채널 순서 변경
    blob = cv2.dnn.blobFromImage(
        frame, scalefactor=1/255.0, size=(416, 416),
        mean=(0, 0, 0), swapRB=True, crop=False
    )

    # 네트워크 입력 설정 후 forward 수행
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes = []        # 검출된 bounding box 저장 [x, y, w, h]
    confidences = []  # confidence 저장
    class_ids = []    # class_id 저장

    # YOLO 출력 해석
    for output in outputs:
        for detection in output:
            # detection[5:]는 클래스별 score
            scores = detection[5:]
            class_id = int(np.argmax(scores))       # 가장 높은 점수의 클래스 선택
            confidence = float(scores[class_id])    # 해당 클래스 confidence

            # confidence가 threshold보다 큰 경우만 사용
            if confidence > CONF_THRESHOLD:
                class_name = classes[class_id]

                # 지정한 클래스만 추적 대상으로 사용
                if class_name not in TARGET_CLASSES:
                    continue

                # YOLO 출력은 중심좌표, 너비, 높이를 비율 형태로 주므로
                # 실제 이미지 크기에 맞게 변환
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # 중심좌표 -> 좌상단 좌표로 변환
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(confidence)
                class_ids.append(class_id)

    # Non-Maximum Suppression 적용
    # 중복으로 잡힌 박스들 중 가장 좋은 박스만 남김
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)

    final_detections = []  # 최종 bbox [x1, y1, x2, y2]
    final_classes = []     # 최종 클래스 이름

    # NMS 결과가 존재하면 최종 detection 구성
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]

            # 이미지 경계를 벗어나지 않도록 보정
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = max(0, x + w)
            y2 = max(0, y + h)

            final_detections.append([x1, y1, x2, y2])
            final_classes.append(classes[class_ids[i]])

    return final_detections, final_classes


# =========================
# 메인 실행
# =========================
def main():
    """
    전체 프로그램 실행 함수

    흐름:
    1. 클래스 이름 로드
    2. YOLO 모델 로드
    3. 비디오 읽기
    4. 프레임마다 객체 검출
    5. SORT로 추적
    6. 박스와 ID를 화면에 출력
    """
    classes = load_classes()          # coco.names 로드
    net, output_layers = load_yolo()  # YOLOv3 로드
    cap = cv2.VideoCapture(VIDEO_PATH)  # 비디오 파일 열기

    # 비디오가 정상적으로 열리지 않으면 종료
    if not cap.isOpened():
        print("비디오 파일을 열 수 없습니다.")
        return

    # SORT tracker 객체 생성
    mot_tracker = Sort(max_age=MAX_AGE, min_hits=MIN_HITS, iou_threshold=IOU_THRESHOLD)

    while True:
        # 비디오에서 한 프레임 읽기
        ret, frame = cap.read()

        # 더 이상 읽을 프레임이 없으면 종료
        if not ret:
            break

        # 현재 프레임에서 YOLO로 객체 검출
        detections, det_classes = detect_objects(frame, net, output_layers, classes)

        # 검출 결과를 SORT에 전달하여 tracking 결과 얻기
        tracks = mot_tracker.update(detections, det_classes)

        # tracking 결과 시각화
        for x1, y1, x2, y2, track_id, class_name in tracks:
            # 객체 bounding box 그리기
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 클래스 이름과 tracker ID 표시
            cv2.putText(
                frame,
                f"{class_name} ID:{track_id}",
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        # 결과 프레임 화면 출력
        cv2.imshow("YOLOv3 + SORT Tracking", frame)

        # 키 입력 대기
        key = cv2.waitKey(1)

        # ESC 키(27)를 누르면 종료
        if key == 27:
            break

    # 자원 해제
    cap.release()
    cv2.destroyAllWindows()


# 현재 파일이 직접 실행될 때만 main() 호출
if __name__ == "__main__":
    main()
```

## 주요 코드 
```python
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
```
→ coco.names 파일을 읽어 각 클래스 이름을 리스트로 저장한다.

```python
class_id = np.argmax(scores)
class_name = classes[class_id]
```
→ YOLO 출력의 class_id를 이용하여 실제 객체 이름으로 변환한다.

```python
TARGET_CLASSES = {"car", "bus", "truck", "motorbike", "person", "bicycle"}
```
→ 모든 객체가 아니라 교통 및 사람 관련 객체만 필터링하여 추적을 수행하였다.

```python
net = cv2.dnn.readNet(WEIGHTS_PATH, CFG_PATH)
```
→ YOLOv3 모델을 로드하는 코드이다.
cfg 파일은 네트워크 구조를 정의하고, weights 파일은 학습된 가중치를 포함한다.

```python
blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416)) 
```
→ 입력 이미지를 YOLO 모델에 맞는 형식으로 변환한다.
픽셀 값을 정규화하고, 416×416 크기로 리사이즈한다.

```python
outputs = net.forward(output_layers)
```
→ YOLOv3 모델을 통해 객체 검출을 수행한다.
각 객체의 위치와 클래스 확률을 반환한다.

```python
iou_matrix[d, t] = iou(det, trk)
```
→ detection과 tracker 간의 IoU를 계산한다.
이 값은 객체 간 유사도를 나타내며, 매칭 과정에 사용된다.

```python
matched_indices = linear_sum_assignment(1 - iou_matrix)
```
→ Hungarian Algorithm을 사용하여 detection과 tracker를 최적으로 매칭한다.
IoU가 높을수록 같은 객체일 가능성이 높다.

```python
self.kf.predict()
```
→ Kalman Filter를 이용하여 다음 프레임에서의 객체 위치를 예측한다.
객체가 잠시 검출되지 않아도 위치를 추정할 수 있다.

```python
self.kf.update(convert_bbox_to_z(bbox))
```
→ 실제 detection 결과를 이용하여 Kalman Filter 상태를 보정한다.

```python
tracks = mot_tracker.update(detections, det_classes)
```
→ 현재 프레임의 detection 결과를 SORT에 입력하여
객체 추적 결과 (bounding box + ID)를 반환한다.

## 결과 화면 

<img width="965" height="596" alt="figure1" src="https://github.com/user-attachments/assets/ba241f4f-a74c-4621-a2a6-db0a5f0e9c7f" />


---
# 2. Mediapipe를 활용한 얼굴랜드마크추출및시각화

## 문제

Mediapipe FaceMesh를 이용하여 실시간 웹캠 영상에서 얼굴을 인식하고, 얼굴의 468개 랜드마크를 추출하여 시각화한다.


## 요구사항

• 웹캠 영상 입력 처리
• Mediapipe FaceMesh 사용
• 얼굴 랜드마크 추출
• 얼굴 윤곽선 시각화
• 실시간 처리

## 전체 코드 (02_cifar10.py)

``` python 
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
```

## 주요 코드 

```python
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)
```
→ FaceMesh 객체를 생성하는 코드이다.
영상 기반 처리이며, 최대 1개의 얼굴을 인식하고 정밀한 landmark를 추출한다.

```python
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
```
→ OpenCV는 BGR 형식을 사용하지만 Mediapipe는 RGB 형식을 사용하므로 변환이 필요하다.

```python
result = face_mesh.process(rgb)
```
→ 입력 이미지에서 얼굴 landmark를 추출한다.
총 468개의 얼굴 특징점을 반환한다.

```python
if result.multi_face_landmarks:
    for face_landmarks in result.multi_face_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            face_landmarks,
            mp_face_mesh.FACEMESH_CONTOURS
        )
```
→ 얼굴이 검출되었는지 확인하는 조건문이다.
검출된 얼굴마다 landmark를 화면에 그린다.
FACEMESH_CONTOURS는 얼굴 윤곽선을 연결하는 방식으로, 눈,코, 입 등의 특징을 시각적으로 구분할 수 있게 해준다.    

```python
cap = cv2.VideoCapture(0)
```
-> 웹캠을 열어 실시간 영상을 입력받는다.

## 결과 

<img width="956" height="764" alt="figure2" src="https://github.com/user-attachments/assets/5a3afb89-465c-4248-a217-a736d1319cb4" />

