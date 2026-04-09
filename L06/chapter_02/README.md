# 02.py 코드 설명: MediaPipe FaceMesh 이미지 랜드마크 검출

이 문서는 02.py를 보면서 설명할 수 있도록 아래 4가지만 담았습니다.

- 코드 설명
- 전체 흐름
- 개념 설명
- 결과물 설명

---

## 1. 코드 설명

02.py는 정적 이미지에서 얼굴 랜드마크를 검출해 점/연결선으로 시각화하는 코드입니다. MediaPipe의 두 경로(legacy `solutions.face_mesh` 또는 `tasks.FaceLandmarker`) 중 가능한 방식을 자동으로 선택합니다.

### 1-1. 라이브러리와 환경 변수

코드 시작에서 로그를 줄이기 위한 환경 변수를 설정합니다.

- `TF_CPP_MIN_LOG_LEVEL=2`
- `TF_ENABLE_ONEDNN_OPTS=0`
- `GLOG_minloglevel=2`

이후 사용 라이브러리:

- `cv2`: 이미지 로드/표시/그리기
- `mediapipe`: 얼굴 랜드마크 검출
- `numpy`: 한글 경로 우회 로딩(`np.fromfile`)용
- `urllib.request.urlretrieve`: task 모델 자동 다운로드
- `os`: 경로 조합 및 파일 존재 확인

### 1-2. 한글 경로 안전 로딩: `load_image_unicode_safe`

`cv2.imread()`로 먼저 읽고 실패하면 Windows 한글 경로 호환 우회를 시도합니다.

1. `np.fromfile(path, dtype=np.uint8)`로 파일 바이트 읽기
2. `cv2.imdecode(..., cv2.IMREAD_COLOR)`로 이미지 디코딩

즉, 일반 로딩 실패 시에도 한글 파일명 이미지(`수지.png`)를 읽을 수 있게 만든 함수입니다.

### 1-3. 검출기 준비: `get_face_landmarker`

MediaPipe 사용 가능 API를 확인해서 분기합니다.

- `mp.solutions.face_mesh` 사용 가능하면 `mode="solutions"` 반환
- 아니면 `tasks` API를 사용

`tasks` 경로에서는 `face_landmarker.task` 파일이 없을 경우 자동 다운로드 후 `FaceLandmarker`를 생성합니다.

핵심 옵션:

- `running_mode=IMAGE` (정적 이미지 모드)
- `num_faces=1` (최대 1명 얼굴)

### 1-4. 메인 처리: `main`

#### 입력 이미지 경로

현재 파일 기준 디렉터리에서 `수지.png`를 읽습니다.

- 실패 시 오류 메시지 출력 후 종료

#### 검출 모드별 처리

A) `solutions` 모드

1. BGR -> RGB 변환
2. `FaceMesh.process()` 실행
3. 얼굴 랜드마크가 있으면
   - `FACEMESH_TESSELATION` 연결선 그리기
   - 각 랜드마크 점(빨간색) 표시
   - "FaceMesh landmarks detected" 텍스트 출력
4. 없으면 "No face landmarks detected" 출력

B) `tasks` 모드

1. BGR -> RGB 변환 후 `mp.Image` 생성
2. `detector["landmarker"].detect()` 실행
3. 랜드마크가 있으면 점을 찍고 포인트 개수 텍스트 출력
4. 없으면 미검출 텍스트 출력
5. 마지막에 `landmarker.close()`로 리소스 해제

#### 결과 표시

- 창 이름: `FaceMesh (Image Mode)`
- ESC 키 입력 시 종료

---

## 2. 전체 흐름

02.py 실행 순서는 다음과 같습니다.

1. 로그 억제 환경 변수를 설정합니다.
2. 이미지를 안전하게 읽는 함수와 랜드마커 생성 함수를 준비합니다.
3. `main()`에서 `수지.png`를 로드합니다.
4. MediaPipe 검출기(`solutions` 또는 `tasks`)를 초기화합니다.
5. 얼굴 랜드마크를 검출합니다.
6. 랜드마크 점/연결선을 출력 이미지에 그립니다.
7. 상태 문구를 오버레이합니다.
8. 결과 창을 띄우고 ESC 입력까지 대기합니다.
9. 창/리소스를 정리하고 종료합니다.

요약하면, 입력 준비 -> 랜드마크 검출 -> 시각화 -> 화면 출력/종료 흐름입니다.

---

## 3. 개념 설명

### Face Landmark

얼굴의 주요 위치(눈, 코, 입, 윤곽 등)를 점 좌표 집합으로 추정하는 기술입니다.

### MediaPipe `solutions` vs `tasks`

- `solutions.face_mesh`: 기존 간편 API
- `tasks.FaceLandmarker`: 최신 task API

코드는 두 방식 중 가능한 경로를 자동 선택해 호환성을 높입니다.

### 정규화 좌표

MediaPipe 랜드마크는 보통 0~1 범위의 정규화 좌표로 제공됩니다. 코드에서 `x * width`, `y * height`로 픽셀 좌표로 변환합니다.

### Tesselation 연결선

얼굴 표면 메시를 삼각형 연결선으로 그려 랜드마크 구조를 직관적으로 보여줍니다.

### 한글 경로 우회 로딩

Windows 환경에서 `cv2.imread`가 한글 경로를 못 읽는 경우를 대비해 `np.fromfile + cv2.imdecode` 방식을 사용합니다.

---

## 4. 결과물 설명

이 코드는 파일 저장보다 화면 시각화를 중심으로 동작합니다.

### 화면 결과

![alt text](image.png)

- 원본 이미지 위에 얼굴 랜드마크 점(빨간색)
- (`solutions` 모드일 때) 랜드마크 연결선(초록색)
- 상태 텍스트
  - `FaceMesh landmarks detected`
  - 또는 `No face landmarks detected`

### 콘솔 결과

- 이미지 로드 실패 시 원인 메시지 출력
- `tasks` 모델이 없을 때 최초 1회 다운로드 메시지 출력
- `ESC 키를 누르면 종료됩니다.` 안내 출력

### 사용 파일

- 입력 이미지: `수지.png` (같은 폴더)
- 모델 파일: `face_landmarker.task` (없으면 자동 다운로드)

해석 포인트:

- 얼굴에 점이 고르게 찍히면 검출이 정상 동작한 상태입니다.
- 미검출 메시지가 나오면 이미지 품질, 얼굴 방향, 가림 여부를 확인해야 합니다.
- `tasks` 모드 텍스트의 포인트 개수로 검출 포맷이 기대대로 동작하는지 확인할 수 있습니다.

---

## 5. 한 줄 정리

02.py는 정적 이미지에서 MediaPipe FaceMesh 랜드마크를 검출해 점/연결선으로 시각화하고, ESC 입력까지 결과 창으로 보여주는 코드입니다.
