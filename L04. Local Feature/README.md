# L04. Local Feature

이 실습은 SIFT를 이용해 아래 3가지를 순서대로 익히는 구성입니다.

1. 한 장의 이미지에서 특징점 찾기
2. 두 이미지 사이에서 같은 지점 매칭하기
3. 매칭 결과로 이미지 정렬(정합)하기

핵심만 기억하면 흐름은 아래 한 줄입니다.

특징 추출 -> 특징 비교(매칭) -> 좋은 매칭만 선택 -> 기하 변환 계산 -> 정렬 결과 확인

---

## 파일 구성

- `chapter_01/01.py`: 단일 이미지 SIFT 특징점 검출/시각화
- `chapter_02/02.py`: 두 이미지 SIFT 특징점 매칭
- `chapter_03/03.py`: SIFT + 호모그래피 기반 이미지 정합

---

## 먼저 알아두면 좋은 개념

### SIFT란?
- 이미지에서 눈에 띄는 점(코너, 질감이 강한 부분)을 찾는 방법입니다.
- 크기 변화, 회전, 밝기 변화에 비교적 강합니다.

### Keypoint와 Descriptor
- Keypoint: 점의 위치 + 크기 + 방향 정보
- Descriptor: 그 점 주변 모양을 숫자 벡터로 표현한 값
- 실제 매칭은 Descriptor 간 거리 비교로 수행됩니다.

### 왜 ratio test를 쓰나?
- 가장 가까운 후보와 두 번째 후보를 비교해
- "확실히 더 가까운 매칭"만 남기기 위해 사용합니다.

### 왜 RANSAC을 쓰나?
- 틀린 매칭(이상치)이 섞여 있어도
- 최대한 안정적으로 변환 행렬(호모그래피)을 추정하기 위해 사용합니다.

---

## 1) chapter_01/01.py

### 이 파일에서 하는 일
- `mot_color70.jpg`에서 SIFT 특징점을 찾고
- 위치/크기/방향을 시각화해 보여줍니다.

### 꼭 봐야 할 핵심 코드

#### 1. SIFT 생성
- `sift = cv.SIFT_create(nfeatures=500)`
- `nfeatures`는 특징점 최대 개수입니다.
  - 너무 많으면 화면이 복잡해지고
  - 너무 적으면 중요한 점이 빠질 수 있습니다.

#### 2. 특징점 계산
- `keypoints, descriptors = sift.detectAndCompute(gray, None)`
- 여기서 이후 단계에 필요한 정보가 모두 만들어집니다.

#### 3. 특징점 시각화
- `cv.drawKeypoints(..., flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)`
- RICH 옵션 덕분에
  - 원 크기로 스케일
  - 방향 표시로 각도
  를 같이 확인할 수 있습니다.

### 나머지 코드는?
- 경로 탐색 함수는 파일 위치 유연성을 위한 보조 코드입니다.
- matplotlib는 결과를 보기 좋게 출력하기 위한 부분입니다.

### 코드 흐름
1. `resolve_image_path()`로 입력 이미지 경로를 찾습니다.
2. `cv.imread()`로 이미지를 읽고, 실패하면 예외 처리합니다.
3. `cv.cvtColor(..., cv.COLOR_BGR2GRAY)`로 그레이스케일로 변환합니다.
4. `cv.SIFT_create()` + `detectAndCompute()`로 keypoint/descriptor를 계산합니다.
5. `cv.drawKeypoints()`로 특징점을 시각화 이미지에 그립니다.
6. 콘솔에 특징점 개수/디스크립터 shape를 출력합니다.
7. matplotlib에서 원본/결과를 나란히 표시하고 종료합니다.

---

## 2) chapter_02/02.py

### 이 파일에서 하는 일
- `mot_color70.jpg`와 `mot_color83.jpg`를 비교해
- 서로 대응되는 특징점을 선으로 연결해 보여줍니다.

### 꼭 봐야 할 핵심 코드

#### 1. 두 이미지에서 SIFT 추출
- `kp1, desc1 = sift.detectAndCompute(gray1, None)`
- `kp2, desc2 = sift.detectAndCompute(gray2, None)`
- 두 이미지 모두에서 descriptor를 만들어야 매칭이 가능합니다.

#### 2. BFMatcher 매칭
- `bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)`
- `matches = bf.match(desc1, desc2)`
- SIFT는 L2 거리 기준이 일반적입니다.
- `crossCheck=True`는 양방향으로 일치하는 매칭만 남겨서 단순하지만 꽤 안정적입니다.

#### 3. FLANN + ratio test (선택)
- 코드 안의 `match_with_flann_ratio_test()`는 더 정교한 방식입니다.
- `knnMatch(..., k=2)` 후
  `m.distance < ratio * n.distance`를 만족하는 매칭만 사용합니다.

#### 4. 시각화
- `cv.drawMatches(...)`로 두 이미지 사이 대응점을 선으로 표시합니다.
- 선이 자연스럽게 연결될수록 매칭 품질이 좋은 편입니다.

### 나머지 코드는?
- `max_draw`는 화면 가독성을 위한 출력 제한입니다.
- 경로/출력 코드는 chapter_01과 같은 보조 역할입니다.

### 코드 흐름
1. 두 입력 이미지 경로를 찾고 `cv.imread()`로 로드합니다.
2. 두 이미지를 각각 그레이스케일로 변환합니다.
3. SIFT로 각 이미지의 keypoint/descriptor를 계산합니다.
4. 매칭 방식을 선택합니다.
  - BF: `BFMatcher(..., crossCheck=True)`
  - FLANN: `knnMatch` + ratio test
5. 거리 기준으로 정렬된 매칭 중 상위 일부(`max_draw`)만 선택합니다.
6. `cv.drawMatches()`로 매칭 선을 그린 결과 이미지를 생성합니다.
7. 매칭 통계를 출력하고 matplotlib로 최종 결과를 확인합니다.

---

## 3) chapter_03/03.py

### 이 파일에서 하는 일
- SIFT 매칭 결과를 이용해 호모그래피를 구하고
- 한 이미지를 다른 이미지 좌표계로 변환해 정렬합니다.

### 꼭 봐야 할 핵심 코드 (가장 중요)

#### 1. 좋은 매칭만 추리기
- `knn_matches = bf.knnMatch(desc1, desc2, k=2)`
- `if m.distance < ratio * n.distance: good_matches.append(m)`
- 이 단계가 정합 품질을 크게 좌우합니다.

#### 2. 호모그래피 계산
- `H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)`
- `H`는 두 이미지 사이의 평면 변환 관계입니다.
- `mask`는 inlier 여부를 알려주므로, 신뢰 매칭만 다시 시각화할 수 있습니다.

#### 3. 이미지 변환(워핑)
- `warped_img1 = cv.warpPerspective(img1, H, (pano_w, pano_h))`
- chapter_03에서 실제 정합이 일어나는 핵심 줄입니다.
- 캔버스를 `(w1 + w2, max(h1, h2))`로 크게 잡아 결과가 잘리지 않게 처리합니다.

#### 4. 결과 확인
- 왼쪽: 변환된 이미지 + 기준 이미지 배치(aligned)
- 오른쪽: inlier 중심 매칭 시각화
- 두 화면을 같이 보면
  - 정렬이 잘 되었는지
  - 어떤 매칭이 신뢰 가능한지
  를 동시에 판단할 수 있습니다.

### 나머지 코드는?
- `pick_second_image()`는 파일명 차이(img2/imag2)를 흡수하는 보조 로직입니다.
- 예외 처리는 디버깅 편의를 위한 안전장치입니다.

### 코드 흐름
1. 기준 이미지(`img1`)와 비교 이미지를 로드합니다.
2. 두 이미지를 그레이스케일로 변환하고 SIFT 특징을 추출합니다.
3. `knnMatch(k=2)`로 매칭 후보를 찾고 ratio test로 좋은 매칭만 남깁니다.
4. 좋은 매칭 좌표(`src_pts`, `dst_pts`)를 만들고 `findHomography(..., RANSAC)`를 수행합니다.
5. 계산된 호모그래피 `H`로 `warpPerspective()`를 적용해 이미지1을 변환합니다.
6. 변환된 결과 위에 기준 이미지(img2)를 배치해 정렬 상태를 만듭니다(`aligned`).
7. RANSAC 마스크가 있으면 inlier만 추려 매칭 시각화를 생성합니다.
8. 왼쪽(정렬 결과)과 오른쪽(매칭 결과)을 함께 출력해 품질을 확인합니다.

---

## 전체 파이프라인 흐름 요약

1. 특징 추출: 각 이미지에서 SIFT keypoint/descriptor 생성
2. 특징 매칭: descriptor 거리 기반으로 대응점 찾기
3. 매칭 정제: ratio test / crossCheck / RANSAC으로 신뢰 매칭 선별
4. 기하 변환: 호모그래피 계산 후 원근 변환
5. 시각 검증: 매칭 선과 정렬 결과를 함께 확인

