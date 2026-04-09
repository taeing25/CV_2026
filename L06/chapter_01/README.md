# 01.py 코드 설명: SORT 알고리즘 기반 다중 객체 추적

이 문서는 01.py를 보면서 설명할 수 있도록 아래 4가지만 담았습니다.

- 코드 설명
- 전체 흐름
- 개념 설명
- 결과물 설명

---

## 1. 코드 설명

01.py는 YOLOv3로 프레임마다 객체를 검출하고, SORT로 프레임 간 같은 객체를 연결해 고유 ID를 유지하는 코드입니다.

### 1-1. 라이브러리 임포트

- opencv-python(cv2): 비디오 입력, DNN 추론, 시각화, 칼만 필터 사용
- numpy: 벡터/행렬 계산
- scipy.optimize.linear_sum_assignment: 헝가리안 알고리즘으로 검출-트랙 매칭

try/except로 필수 패키지 누락 시 실행을 중단하고 설치 안내를 출력합니다.

### 1-2. 입력 옵션과 모델 경로

- parse_args(): 실행 옵션을 정의합니다.
  - --source: 웹캠 번호 또는 비디오 파일
  - --conf-threshold: 검출 신뢰도 임계값
  - --nms-threshold: NMS 임계값
  - --max-age, --min-hits: 트랙 유지 기준
  - --output: 결과 영상 저장 경로
  - --width, --height: 출력 프레임 크기

- resolve_model_paths(): 모델 파일 존재 여부를 확인합니다.
  - L06/yolov3.cfg
  - L06/yolov3.weights

파일이 없으면 예외를 발생시켜 실행을 중지합니다.

### 1-3. 비디오 입력 처리

- open_video_source(source)
  - source가 숫자면 웹캠으로 열고,
  - 실패하면 fallback 영상(slow_traffic_small.mp4)을 시도합니다.
  - 그래도 실패하면 RuntimeError를 발생시킵니다.

### 1-4. YOLO 출력 레이어 준비

- get_output_layer_names(net)
  - OpenCV DNN에서 YOLO의 출력 레이어 이름 목록을 가져옵니다.
  - getUnconnectedOutLayers 형식 차이를 고려해 인덱스를 평탄화 처리합니다.

### 1-5. 객체 검출 함수: detect_objects

프레임 1장을 입력받아 최종 검출 박스를 반환합니다.

동작 순서:

1. 프레임을 blob으로 변환
2. net.forward로 YOLO 추론
3. confidence 임계값 이하 결과 제거
4. NMS로 중복 박스 제거
5. [x1, y1, x2, y2, confidence] 형식으로 반환

반환 타입:

- 검출이 없으면 shape (0, 5) 빈 배열
- 검출이 있으면 float32 배열

### 1-6. 좌표 변환/매칭 함수

- bbox_to_measurement: 박스 좌표를 칼만 필터 측정 벡터(cx, cy, scale, ratio)로 변환
- measurement_to_bbox: 상태 벡터를 화면 좌표 박스로 복원
- iou: 두 박스 IoU 계산
- associate_detections_to_trackers:
  - IoU 행렬 계산
  - cost = 1 - IoU로 변환
  - 헝가리안 알고리즘으로 최적 매칭
  - IoU 임계값 미만 매칭 제거
  - matches, unmatched_detections 반환

### 1-7. KalmanBoxTracker 클래스

단일 객체 트랙을 담당합니다.

- **init**: 7차 상태, 4차 측정 칼만 필터 초기화
- predict(): 다음 상태 예측
- update(bbox): 새 관측값으로 보정
- get_state(): 현재 상태를 박스 좌표로 반환

관리 변수:

- id, age, hits, hit_streak, time_since_update

### 1-8. Sort 클래스

다중 객체 트랙을 관리합니다.

update(detections)에서 수행하는 핵심 단계:

1. 모든 트랙 predict
2. 검출-트랙 매칭
3. 매칭된 트랙 update
4. 미매칭 검출은 새 tracker 생성
5. 안정 트랙만 active_tracks로 반환
6. max_age 초과 트랙 제거

반환 형식:

- [x1, y1, x2, y2, track_id] 배열

### 1-9. 시각화와 메인 루프

- draw_tracks(frame, tracks): 박스와 ID 텍스트를 프레임에 그림
- main(): 전체 파이프라인 실행
  - 인자 파싱
  - YOLO 로드
  - 비디오 입력 열기
  - 프레임 반복
  - detect_objects -> tracker.update -> draw_tracks
  - Frame, Detections, Tracks 정보 표시
  - 화면 출력 및 선택적 파일 저장
  - q 또는 ESC로 종료

---

## 2. 전체 흐름

01.py 전체 실행 순서는 아래와 같습니다.

1. 실행 옵션을 읽습니다.
2. YOLO cfg/weights 경로를 확인합니다.
3. 비디오 소스를 엽니다.
4. YOLO 네트워크와 출력 레이어를 준비합니다.
5. 프레임 반복 루프를 시작합니다.
6. YOLO로 객체를 검출합니다.
7. SORT로 검출 결과를 추적하고 ID를 부여합니다.
8. 프레임에 박스/ID/상태 정보를 그립니다.
9. 화면 출력(및 옵션에 따라 파일 저장) 후 종료 조건을 확인합니다.
10. 리소스를 해제하고 종료합니다.

요약하면 준비 단계 -> 검출+추적 반복 -> 결과 출력/정리 순서입니다.

---

## 3. 개념 설명

### YOLOv3

한 프레임에서 객체 위치와 신뢰도를 빠르게 예측하는 객체 검출 모델입니다.

### NMS (Non-Maximum Suppression)

겹치는 여러 박스 중 신뢰도가 높은 박스만 남겨 중복 검출을 줄이는 기법입니다.

### SORT

프레임별 검출 결과를 연결해 객체 ID를 유지하는 경량 다중 객체 추적 알고리즘입니다.

### 칼만 필터

객체 위치를 예측하고 관측값으로 보정해, 노이즈가 있는 상황에서도 추적을 안정화합니다.

### IoU

두 박스의 겹침 비율입니다. 검출과 트랙의 유사도를 수치화할 때 사용합니다.

### 헝가리안 알고리즘

검출 박스와 트랙의 전체 매칭 비용을 최소화하는 최적 할당 알고리즘입니다.

### max_age / min_hits

- max_age: 검출이 없어도 트랙을 유지할 최대 프레임 수
- min_hits: 화면에 안정 트랙으로 표시하기 위한 최소 누적 검출 횟수

---

## 4. 결과물 설명

이 코드는 실시간 화면 출력이 기본 결과이며, 옵션 설정 시 비디오 파일도 생성합니다.

### 화면에 표시되는 정보

- 객체 바운딩 박스
- 객체 고유 ID (ID n)
- 현재 프레임 번호 (Frame)
- 현재 검출 수 (Detections)
- 현재 추적 수 (Tracks)

### 파일 결과물

- --output 경로를 지정하면 추적 결과 영상(mp4)을 저장합니다.
- --output을 지정하지 않으면 파일 저장 없이 화면 출력만 수행합니다.

### 실행 예시

python 01.py --source 0
python 01.py --source ../slow_traffic_small.mp4
python 01.py --source ../slow_traffic_small.mp4 --output result.mp4

### 해석 포인트

- 동일 객체가 여러 프레임에서 같은 ID를 유지하면 추적이 안정적입니다.
- 잠시 가려진 객체가 재등장했을 때 ID가 유지되면 추적 품질이 좋습니다.
- Detections와 Tracks 차이가 크면 conf-threshold, nms-threshold, max-age 값을 조정해 개선할 수 있습니다.
