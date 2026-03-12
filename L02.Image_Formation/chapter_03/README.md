# 📷 스테레오 비전으로 깊이(거리) 측정하기

> 두 카메라로 찍은 사진을 이용해 각 물체까지의 거리를 자동으로 계산하는 프로그램

---

## 📌 목차

1. [문제 정의 — 왜 이 프로젝트가 필요한가?](#1-문제-정의)
2. [해결 방법 — 어떤 원리로 거리를 구하는가?](#2-해결-방법)
3. [코드 상세 설명](#3-코드-상세-설명)
4. [실행 결과 해석](#4-실행-결과-해석)

---

## 1. 문제 정의

### 🤔 카메라 한 대로 거리를 알 수 있을까?

일반 카메라 사진 한 장만으로는 **물체가 얼마나 멀리 있는지 알 수 없습니다.**  
사진은 3D 세계를 2D로 납작하게 담기 때문에, 깊이(Depth) 정보가 사라집니다.

> 예를 들어, 멀리 있는 큰 나무와 가까이 있는 작은 나무가 사진에서 같은 크기로 찍힐 수 있습니다.

### 👀 그렇다면 사람은 어떻게 거리를 느낄까?

사람의 눈은 두 개이고, 왼쪽 눈과 오른쪽 눈은 약간 다른 위치에서 세상을 봅니다.  
뇌는 두 눈에서 들어오는 **"약간 다른 두 이미지"** 를 비교해서 거리감(입체감)을 느낍니다.

이것을 **양안시차(Binocular Disparity)** 라고 합니다.

```
왼쪽 눈으로 본 모습      오른쪽 눈으로 본 모습
  [물체가 약간 오른쪽]      [물체가 약간 왼쪽]
         ↕ 두 이미지를 비교
      → 얼마나 이동했는지 = Disparity
      → Disparity가 크면 → 가까운 물체
      → Disparity가 작으면 → 먼 물체
```

### 🎯 이 프로젝트의 목표

- 왼쪽/오른쪽 카메라로 찍은 **두 장의 이미지**를 입력으로 받는다
- 각 픽셀의 **Disparity(시차)** 를 계산한다
- Disparity로부터 실제 **Depth(거리, 단위: 미터)** 를 계산한다
- 장면 속 3개 영역(Painting / Frog / Teddy)이 **카메라로부터 얼마나 떨어져 있는지** 비교한다

---

## 2. 해결 방법

### 🔑 핵심 개념: Disparity란?

같은 물체를 왼쪽 카메라와 오른쪽 카메라로 찍으면,  
물체가 두 이미지에서 **가로 방향으로 약간 다른 위치**에 찍힙니다.

```
왼쪽 이미지에서 물체의 x 좌표: 150px
오른쪽 이미지에서 물체의 x 좌표: 130px

Disparity = 150 - 130 = 20px   ← 이 차이가 "시차"
```

**가까운 물체일수록 이 차이(Disparity)가 커집니다.**

---

### 📐 Disparity → Depth 변환 공식

Disparity를 알면 아래 공식으로 실제 거리를 계산할 수 있습니다:

```
        f × B
Z  =  ─────────
          d

Z : 물체까지의 거리 (Depth, 단위: 미터)
f : 카메라의 초점 거리 (Focal Length, 단위: 픽셀)
B : 두 카메라 사이의 거리 (Baseline, 단위: 미터)
d : Disparity 값 (픽셀 단위 시차)
```

> 💡 직관적 이해:  
> d(시차)가 크면 → Z(거리)는 작아짐 → 가까운 물체  
> d(시차)가 작으면 → Z(거리)는 커짐 → 먼 물체

---

### 🛠️ 전체 처리 흐름

```
[왼쪽 이미지]  [오른쪽 이미지]
      ↓               ↓
   그레이스케일 변환 (컬러 → 흑백)
      ↓               ↓
   StereoBM 알고리즘으로 Disparity Map 계산
              ↓
       Z = fB/d 로 Depth Map 계산
              ↓
    ROI(관심 영역)별 평균 값 분석
              ↓
    컬러맵으로 시각화 (가까움=빨강, 멀음=파랑)
```

---

### 🤖 StereoBM 알고리즘이란?

**Block Matching(블록 매칭)** 방식의 스테레오 매칭 알고리즘입니다.

원리를 간단히 설명하면:

1. 왼쪽 이미지의 특정 픽셀 주변 블록(예: 15×15)을 선택
2. 오른쪽 이미지의 같은 행에서 비슷하게 생긴 블록을 찾음
3. 두 블록의 **가로 위치 차이** = Disparity

```
왼쪽 이미지의 블록     오른쪽 이미지에서 탐색
┌─────────┐           ←←←←←←←←←←←←←←←←←←←←
│ 🐸🐸🐸 │   찾기!   [탐색 범위: numDisparities]
│ 🐸🐸🐸 │  ──────▶  가장 비슷한 블록 위치 발견
│ 🐸🐸🐸 │           차이 = Disparity 값
└─────────┘
```

---

## 3. 코드 상세 설명

### 📂 Step 0. 기본 설정

```python
f = 700.0   # 초점 거리: 카메라 렌즈의 특성값 (픽셀 단위)
B = 0.12    # Baseline: 두 카메라 사이의 실제 거리 (12cm = 0.12m)
```

```python
rois = {
    "Painting": (55, 50, 130, 110),   # (x좌표, y좌표, 너비, 높이)
    "Frog":     (90, 265, 230, 95),
    "Teddy":   (310,  35, 115, 90)
}
```
> ROI(Region of Interest): 분석할 관심 영역의 위치와 크기를 픽셀 좌표로 지정합니다.

---

### 🎨 Step 1. 그레이스케일 변환

```python
left_gray  = cv2.cvtColor(left_color,  cv2.COLOR_BGR2GRAY)
right_gray = cv2.cvtColor(right_color, cv2.COLOR_BGR2GRAY)
```

**왜 흑백으로 바꾸나요?**

StereoBM은 픽셀의 **밝기 패턴**만 비교합니다.  
컬러(BGR) 이미지는 채널이 3개(B, G, R)라 연산이 3배 느려지고, 색상 정보가 오히려 방해가 될 수 있습니다.  
흑백으로 변환하면 **빠르고 안정적인** 매칭이 가능합니다.

---

### 🔍 Step 2. Disparity Map 계산

```python
stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
disparity_raw = stereo.compute(left_gray, right_gray)

# StereoBM의 결과는 실제값의 16배로 저장됨 → 16으로 나눠서 실제 disparity로 변환
disparity = disparity_raw.astype(np.float32) / 16.0
```

**파라미터 설명:**

| 파라미터 | 값 | 의미 |
|---|---|---|
| `numDisparities` | 64 | 탐색할 최대 시차 범위 (16의 배수여야 함) |
| `blockSize` | 15 | 매칭에 사용할 블록 크기 (홀수여야 함, 클수록 부드럽지만 디테일 손실) |

**왜 16으로 나누나요?**  
OpenCV의 StereoBM은 정밀도를 높이기 위해 결과를 **정수×16** 형태로 저장합니다.  
예: 실제 disparity가 12.5이면, 200으로 저장 → 200 ÷ 16 = 12.5

---

### 📏 Step 3. Depth Map 계산

```python
valid_mask = disparity > 0   # 유효한 픽셀만 선택 (0 이하는 계산 불가/오류)
depth_map = np.zeros_like(disparity, dtype=np.float32)
depth_map[valid_mask] = (f * B) / disparity[valid_mask]
```

**왜 `disparity > 0`인 픽셀만 쓰나요?**

- `disparity = 0` 이면 공식에서 **0으로 나누기** 가 발생 → 무한대(∞) 오류
- 음수 disparity는 매칭 실패를 의미하는 **무효값**
- 따라서 양수 disparity만 사용해야 신뢰할 수 있는 depth 값을 얻을 수 있습니다

---

### 📊 Step 4. ROI별 평균 분석

```python
for name, (x, y, w, h) in rois.items():
    roi_disp  = disparity[y:y+h, x:x+w]   # 해당 ROI 영역만 잘라냄
    roi_depth = depth_map[y:y+h, x:x+w]

    # disparity > 0인 픽셀만 선택해서 평균 계산
    valid_disp_pixels  = roi_disp[roi_disp > 0]
    valid_depth_pixels = roi_depth[roi_depth > 0]

    avg_disp  = np.mean(valid_disp_pixels)
    avg_depth = np.mean(valid_depth_pixels)
```

이미지에서 ROI를 자르는 방식:

```
전체 이미지 배열
┌──────────────────────────────┐
│                              │
│     (x, y)                   │
│       ┌──────────┐           │
│       │  ROI 영역 │  ← h 높이 │
│       │          │           │
│       └──────────┘           │
│          ← w 너비 →           │
└──────────────────────────────┘

코드: image[y : y+h, x : x+w]
      (행 범위)  (열 범위)
```

---

### 🎨 Step 5 & 6. 시각화

**Disparity 시각화 (빨강=가까움, 파랑=멀음)**

```python
# 유효하지 않은 픽셀은 NaN으로 처리 (시각화에서 제외)
disp_tmp[disp_tmp <= 0] = np.nan

# 상위/하위 5% 극단값 제거 후 0~255 범위로 정규화
d_min = np.nanpercentile(disp_tmp, 5)
d_max = np.nanpercentile(disp_tmp, 95)
disp_scaled = (disp_tmp - d_min) / (d_max - d_min)   # 0.0 ~ 1.0

# JET 컬러맵 적용: 낮은값=파랑, 높은값=빨강
disparity_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
```

**Depth 시각화 (빨강=가까움, 파랑=멀음)**

```python
# depth는 클수록 멀기 때문에 1에서 빼서 반전
depth_scaled = 1.0 - depth_scaled
# 이제 가까운(작은 depth) → 높은 값 → JET에서 빨강으로 표현됨
```

> 💡 반전이 필요한 이유:  
> JET 컬러맵은 값이 클수록 빨강입니다.  
> Disparity는 클수록 가까우니까 그대로 써도 되지만,  
> Depth는 클수록 **멀기** 때문에 반전시켜야 가까운 것이 빨강으로 나옵니다.

---

### 💾 Step 8 & 9. 저장 및 화면 출력

```python
# 4개 결과 이미지 저장
cv2.imwrite("outputs/left_roi.png",       left_vis)
cv2.imwrite("outputs/right_roi.png",      right_vis)
cv2.imwrite("outputs/disparity_map.png",  disparity_color)
cv2.imwrite("outputs/depth_map.png",      depth_color)

# 화면에 각각 표시
cv2.imshow("Left ROI",       left_vis)
cv2.imshow("Right ROI",      right_vis)
cv2.imshow("Disparity Map",  disparity_color)
cv2.imshow("Depth Map",      depth_color)
cv2.waitKey(0)          # 키 입력 대기
cv2.destroyAllWindows() # 창 모두 닫기
```

---

## 4. 실행 결과 해석

### 📋 터미널 출력 

```
==================================================
ROI            Avg Disparity   Avg Depth (m)
--------------------------------------------------
Painting               19.06          4.4248
Frog                   33.60          2.5119
Teddy                  22.42          3.8926
==================================================

가장 가까운 ROI : Frog  (avg disparity = 33.60)
가장 먼    ROI : Painting  (avg disparity = 19.06)


### 🗺️ 컬러맵 시각화 해석

| 색상 | 의미 |
|------|------|
| 🔴 빨강 | 가까운 물체 (Disparity 큼 / Depth 작음) |
| 🟡 노랑/초록 | 중간 거리 |
| 🔵 파랑 | 먼 물체 (Disparity 작음 / Depth 큼) |
| ⬛ 검정 | 유효하지 않은 픽셀 (매칭 실패) |

### 💡 결론적으로...

```
Disparity 크다 → 시차가 크다 → 가까운 물체
Disparity 작다 → 시차가 작다 → 먼 물체

Depth 작다 → 카메라에 가까움
Depth 크다 → 카메라에서 멀리 있음
```

