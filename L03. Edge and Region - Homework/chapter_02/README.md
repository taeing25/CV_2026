# 📐 허프 변환을 이용한 직선 검출

> 캐니 에지 검출 후 허프 변환으로 이미지에서 직선을 검출하는 프로그램

---

## 📌 문제 정의

### 직선 검출이란?
에지 맵에서 **일직선상으로 연결된 점들의 집합**을 찾아내는 작업

### 목표
- 캐니 에지로 에지 맵 생성
- 허프 변환으로 직선 검출
- 검출된 직선을 원본 이미지에 그리기

---

## 🔍 해결 방법

### 캐니 에지 검출
이미지의 밝기 변화량이 급격한 부분을 검출하여 0과 255로 변환

```
원본 이미지 → 가우시안 블러 → Gradient 계산 → Non-Maximum Suppression → 히스테리시스
           (노이즈 제거)    (밝기 변화)      (에지 얇게)           (강한 에지 추적)
```

### 허프 변환 (Hough Transform)
에지 맵에서 직선, 원 등 특정 형태를 검출하는 알고리즘

```
edges 배열 (이진 이미지: 0과 255)
        ↓
흰색 픽셀(255)들을 찾음
        ↓
"이 흰 점들이 일렬로 정렬되어 있나?"를 투표
        ↓
많은 표를 받은 점들의 집합 = 직선!
        ↓
직선의 시작점(x1,y1)과 끝점(x2,y2) 반환
```

---

## 1️⃣ 이미지 로드 및 그레이스케일 변환

```python
img = cv.imread('L03. Edge and Region - Homework/dabo.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
```

**설명:**
- `cv.imread()`: 원본 이미지 로드 (BGR 형식)
- `cv.cvtColor()`: 컬러 이미지를 그레이스케일로 변환 (어두운 값과 밝은 값으로 표현)
- 허프 변환은 그레이스케일에서 작동

---

## 2️⃣ 캐니 에지 검출

```python
edges = cv.Canny(gray, 150, 250)
```

**설명:**
- `threshold1=150`: 하단 값 (이 미만은 확실한 배경)
- `threshold2=250`: 상단 값 (이 이상은 확실한 에지)
- `150~250 사이`: 주변에 확실한 에지가 있으면 포함
- 결과: 결과: 이진 이미지 (0=배경, 255=에지)

---

## 3️⃣ 허프 변환으로 직선 검출

```python
lines = cv.HoughLinesP(edges, rho=1, theta=np.pi/180, 
                       threshold=70, minLineLength=100, maxLineGap=8)
```

**설명:**
- `rho=1`: 거리 해상도 (1픽셀 간격)
- `theta=np.pi/180`: 각도 해상도 (1도 간격)
- `threshold=70`: 직선으로 인정하는 최소 투표 수 (높을수록 강한 직선만 선택)
- `minLineLength=100`: 최소 직선 길이 (주요 구조선만 추출)
- `maxLineGap=8`: 직선으로 연결할 최대 간격 (끊긴 선 연결)

---

## 4️⃣ 검출된 직선 그리기

```python
line_img = img.copy()
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
```

**설명:**
- `img.copy()`: 원본 보존을 위해 복사본 생성
- `line[0]`: line 배열에서 좌표 추출 (x1, y1, x2, y2)
- `cv.line()`: 이미지에 직선 그리기
  - `(0, 0, 255)`: BGR 형식의 빨간색
  - `2`: 선의 두께 (픽셀)

---

## 5️⃣ 결과 시각화

```python
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv.cvtColor(line_img, cv.COLOR_BGR2RGB))
plt.title('Detected Lines (Red)')
plt.axis('off')

plt.tight_layout()
plt.show()
```

**설명:**
- `figsize=(12, 6)`: 그림 크기 (가로 12, 세로 6)
- `subplot(1, 2, 1)`: 1행 2열, 첫 번째 위치 (원본)
- `subplot(1, 2, 2)`: 1행 2열, 두 번째 위치 (검출 결과)
- `cv.cvtColor()`: BGR을 RGB로 변환 (matplotlib은 RGB 형식)
- 원본과 검출 결과를 나란히 비교

---

## 📊 파라미터 조정 가이드

| 파라미터 | 역할 | 조정 시 |
|---------|------|--------|
| **threshold** | 직선 판정 최소 투표 수 | ↑ 올리면 강한 직선만 |
| **minLineLength** | 최소 직선 길이 | ↑ 올리면 긴 선만 |
| **maxLineGap** | 직선 연결 최대 간격 | ↓ 내리면 덜 연결 |

---

## 💡 좋은 결과를 위한 팁

1️⃣ Canny 임계값 조정
   - 너무 낮으면 (100, 200): 노이즈까지 포함
   - 적당함 (150, 250): 권장
   - 너무 높으면 (200, 300): 중요한 에지 손실

2️⃣ HoughLinesP 파라미터 조정
   - threshold 70~90: 주요 직선만
   - minLineLength 80~150: 이미지 크기에 맞춰 조정

3️⃣ 테스트하며 최적값 찾기
   - 다양한 threshold와 minLineLength 조합 시도

---
