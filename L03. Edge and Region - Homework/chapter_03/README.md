# 🎯 GrabCut을 이용한 배경 제거

> GrabCut 알고리즘으로 이미지에서 객체를 분리하고 배경을 제거하는 프로그램

---

## 📌 문제 정의

### 배경 제거란?
이미지에서 **원하는 객체(전경)만 남기고 배경을 투명하게 또는 검은색으로 제거**하는 작업

### 목표
- 사용자가 지정한 영역에서 객체와 배경 자동 분리
- GrabCut 알고리즘으로 정확한 세그멘테이션
- 배경을 제거한 깔끔한 결과 이미지 생성

---

## 🔍 해결 방법

### GrabCut 알고리즘
사용자가 지정한 **사각형 영역** 내에서 그래프 컷(Graph Cut) 알고리즘을 사용하여 전경과 배경을 자동으로 구분

```
원본 이미지 + 사각형 영역 (rect)
           ↓
배경/전경 색상 특성 학습
           ↓
반복 계산 (15회)
           ↓
각 픽셀을 배경(0) 또는 전경(1)으로 분류
           ↓
마스크 생성 (0 = 배경, 1 = 객체)
           ↓
노이즈 제거 + 경계 부드럽게
           ↓
최종 배경 제거 이미지
```

---

## 1️⃣ 이미지 로드 및 초기화

```python
img = cv.imread('L03. Edge and Region - Homework/coffee cup.jpg')

mask = np.zeros(img.shape[:2], np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)
```

**설명:**
- `mask`: 이미지 크기와 같은 0 배열 (GrabCut 결과를 저장)
- `bgdModel`, `fgdModel`: GrabCut 알고리즘 내부에서 사용하는 배경/전경 학습 모델

---

## 2️⃣ 사각형 영역 설정 및 GrabCut 실행

```python
rect = (150, 50, 1000, 850)
cv.grabCut(img, mask, rect, bgdModel, fgdModel, 15, cv.GC_INIT_WITH_RECT)
```

**설명:**
- `rect = (x, y, width, height)`: 객체를 포함할 사각형 (배경을 줄일수록 정확함)
- `iterCount=15`: 반복 횟수 (많을수록 정밀하지만 느림, 10~15 권장)
- 결과: mask에 0(배경), 1(전경), 2(아마 배경), 3(아마 전경) 저장

---

## 3️⃣ 마스크 변환 및 노이즈 제거

```python
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
mask2 = cv.morphologyEx(mask2, cv.MORPH_OPEN, kernel)    # 작은 노이즈 제거
mask2 = cv.morphologyEx(mask2, cv.MORPH_CLOSE, kernel)   # 구멍 채우기
```

**설명:**
- **마스크 변환**: 2와 0(배경)을 모두 0으로, 1과 3(전경)을 1로 통일
- **MORPH_OPEN**: 작은 흰 점(노이즈) 제거
- **MORPH_CLOSE**: 객체 내부의 검은 구멍 채우기

---

## 4️⃣ 가장 큰 객체만 추출

```python
contours, _ = cv.findContours(mask2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
if contours:
    largest_contour = max(contours, key=cv.contourArea)
    mask2 = np.zeros_like(mask2)
    cv.drawContours(mask2, [largest_contour], 0, 1, -1)
```

**설명:**
- 마스크에서 모든 객체 윤곽선 찾기
- 가장 큰 객체만 선택 (배경의 작은 노이즈 제거)
- 선택된 객체만 흰색(1)으로 그리기

---

## 5️⃣ 경계 부드럽게 처리

```python
mask2 = cv.GaussianBlur(mask2, (5, 5), 0)
mask2 = np.where(mask2 > 0.5, 1, 0).astype('uint8')
```

**설명:**
- `GaussianBlur`: 마스크 경계를 부드럽게 (톱니 모양 제거)
- 블러된 값이 0.5 이상이면 1(전경), 아니면 0(배경)으로 변환
- 결과: 자연스러운 객체 경계

---

## 6️⃣ 배경 제거 및 결과 표시

```python
img_result = img * mask2[:, :, np.newaxis]

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.title('Original')
plt.subplot(1, 3, 3)
plt.imshow(cv.cvtColor(img_result, cv.COLOR_BGR2RGB))
plt.title('Result')
plt.show()
```

**설명:**
- `img * mask2[:, :, np.newaxis]`: 마스크를 곱해서 배경(0) 부분을 검은색으로
- `[:, :, np.newaxis]`: 채널 차원 추가 (3채널과 맞춤)
- 결과: 원본 이미지와 배경 제거된 이미지 비교

---
<img width="1500" height="500" alt="Figure_1_수정" src="https://github.com/user-attachments/assets/056e90a0-0cdc-4292-bc30-649b65c7e1cf" />


