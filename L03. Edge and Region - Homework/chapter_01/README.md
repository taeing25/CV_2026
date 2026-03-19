# 📐 소벨 필터를 이용한 에지 검출

> 소벨 필터를 사용하여 이미지에서 에지를 검출하고 에지 강도를 시각화하는 프로그램
---

## 📌 문제 정의

### 에지 검출이란?
이미지에서 **밝기 변화가 급격한 부분(경계선)**을 찾아내는 작업

### 목표
- 이미지의 x축, y축 방향 미분 계산
- 소벨 필터로 각 방향의 에지 검출
- 에지 강도 계산 및 시각화

---

## 🔍 해결 방법

### 소벨 필터 (Sobel Filter)
이미지의 미분을 계산하는 필터로, 밝기 변화량을 구하여 에지를 검출

```
원본 이미지
        ↓
그레이스케일 변환
        ↓
x축 방향 미분 (소벨)  ← 수평선(—) 검출
y축 방향 미분 (소벨)  ← 수직선(|) 검출
        ↓
에지 강도 계산: √(Sobel_x² + Sobel_y²)
        ↓
에지 강도 이미지 생성
```

### 에지 강도 계산
- **Sobel_x**: x축 방향의 밝기 변화 (dx=1, dy=0)
- **Sobel_y**: y축 방향의 밝기 변화 (dx=0, dy=1)
- **Magnitude**: √(Sobel_x² + Sobel_y²)

---

## 1️⃣ 이미지 로드 및 에러 처리

```python
img_path = 'L03. Edge and Region - Homework/edgeDetectionImage.jpg'
img = cv.imread(img_path)

if img is None:
    print("이미지를 불러올 수 없습니다. 경로를 확인해주세요.")
```

**설명:**
- `cv.imread()`: 이미지 파일을 메모리에 로드 (BGR 형식)
- `if img is None`: 파일을 찾을 수 없으면 에러 메시지 출력
- 올바른 경로 확인이 중요함

---

## 2️⃣ 그레이스케일 변환

```python
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
```

**설명:**
- 컬러 이미지를 그레이스케일로 변환 (0~255의 단일 채널)
- 소벨 필터는 그레이스케일에서만 작동
- 계산 속도 향상 및 밝기 변화 분석 용이

---

## 3️⃣ x축, y축 방향 에지 검출

```python
sobel_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)   # 세로 에지
sobel_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)   # 가로 에지
```

**설명:**
- `cv.CV_64F`: 64비트 부동소수점 (음수값 보존, 정확도 높음)
- `dx=1, dy=0`: x축 방향 미분 → 수직 에지 검출
- `dx=0, dy=1`: y축 방향 미분 → 수평 에지 검출
- `ksize=3`: 3×3 커널 크기 (권장)

---

## 4️⃣ 에지 강도(Magnitude) 계산

```python
mag = cv.magnitude(sobel_x, sobel_y)
```

**설명:**
- 두 벡터(sobel_x, sobel_y)의 크기 계산
- 공식: √(sobel_x² + sobel_y²)
- 모든 방향의 에지를 합친 강도값 생성

---

## 5️⃣ 절댓값 변환 및 스케일링

```python
sobel_combined = cv.convertScaleAbs(mag)
```

**설명:**
- 음수값을 절댓값으로 변환 (에지의 방향 정보 보존)
- 자동으로 0~255 범위로 스케일링
- `uint8` 형식으로 변환하여 시각화 가능하게 만듦

---

## 6️⃣ 결과 시각화

```python
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(sobel_combined, cmap='gray')
plt.title('Sobel Edge Magnitude')
plt.axis('off')

plt.tight_layout()
plt.show()
```

**설명:**
- `figsize=(12, 6)`: 그림 크기 (가로 12, 세로 6)
- `subplot(1, 2, 1)`: 1행 2열, 첫 번째 위치 (원본)
- `subplot(1, 2, 2)`: 1행 2열, 두 번째 위치 (에지 결과)
- `cv.cvtColor()`: BGR을 RGB로 변환 (matplotlib은 RGB 형식)
- `cmap='gray'`: 그레이스케일로 표시

---

## 📊 파라미터 조정 가이드

| 파라미터 | 역할 | 설명 |
|---------|------|------|
| **dx, dy** | 미분 방향 | dx=1 (x축), dy=1 (y축) |
| **ksize** | 커널 크기 | 3 (권장), 5, 7... (홀수만) |
| **cv.CV_64F** | 데이터 타입 | 부동소수점 (음수 표현 가능) |

---

## 💡 좋은 결과를 위한 팁

1️⃣ **ksize 선택**
   - ksize=3: 민감한 에지 검출 (권장)
   - ksize=5: 균형잡힌 결과
   - ksize=7: 강한 노이즈 제거

2️⃣ **데이터 타입 선택**
   - cv.CV_64F 필수 (음수값 손실 방지)

3️⃣ **결과 해석**
   - 밝은 부분: 강한 에지
   - 어두운 부분: 약한 에지 또는 배경

---

### ⚠️ ksize는 홀수만 가능
```python
# 유효한 값
cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)  # ✅
cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=5)  # ✅

# 오류 발생
# cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=4)  # ❌ 짝수 불가
```

### ⚠️ BGR과 RGB의 차이
```python
# OpenCV는 BGR 형식으로 로드하므로 matplotlib에 표시할 때 변환 필수
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))  # ✅ 올바른 색상
# plt.imshow(img)  # ❌ 빨강과 파랑이 바뀜
```

### 💡 x축과 y축 방향의 차이
```python
sobel_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)  # 세로 에지 검출
sobel_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)  # 가로 에지 검출

# x축 미분 → 수직 방향의 변화 → 수평선(—) 검출
# y축 미분 → 수평 방향의 변화 → 수직선(|) 검출
```

---

## 📸 실행 결과 비교

| 커널 크기 | 민감도 | 노이즈 | 사용 시기 |
|---------|--------|--------|---------|
| ksize=1 | 매우 높음 | 많음 | 세부 정보 필요 시 |
| **ksize=3** | **중간** | **적음** | **일반적으로 권장** ✅ |
| ksize=5 | 낮음 | 거의 없음 | 노이즈 제거 중요 시 |

---
