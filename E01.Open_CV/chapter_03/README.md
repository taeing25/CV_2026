# 마우스로 영역 선택 및 저장 프로그램

마우스로 이미지에서 관심 영역(ROI)을 선택하고, 별도 창에 표시하거나 파일로 저장할 수 있는 프로그램입니다.

드래그로 사각형을 그려 영역을 선택하고, 키보드로 리셋하거나 저장할 수 있습니다.

---

## 과제 설명
<img width="641" height="413" alt="image" src="https://github.com/user-attachments/assets/5e1d358f-7bfd-4cd5-8bfa-e90e8b459ef6" />
---

### 필요한 라이브러리

```python
import cv2 as cv
import sys
```

**라이브러리 설명:**
- `cv2 as cv` : OpenCV - 이미지를 다루고 마우스 이벤트를 처리하는 도구
- `sys` : 프로그램을 종료할 때 사용하는 도구

---

## 1단계: 이미지 로드 및 전역 변수 초기화

```python
img = cv.imread('soccer.jpg')
if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

img_copy = img.copy()
roi = None
ix, iy = -1, -1
drawing = False
```

**하는 일:**
- `cv.imread('soccer.jpg')` : 같은 폴더에서 soccer.jpg 파일을 불러옴
- `if img is None:` : 파일이 없으면 프로그램 종료
- `img_copy = img.copy()` : 원본 이미지를 복사 (사각형을 그릴 용도)
- `roi = None` : 선택된 영역을 저장할 변수 (초기값 없음)
- `ix, iy = -1, -1` : 마우스 시작 좌표 (초기값 -1)
- `drawing = False` : 마우스 클릭 상태 (초기값 False)

프로그램에 필요한 이미지와 변수를 준비합니다.

---

## 2단계: 마우스 콜백 함수 정의

```python
def draw_roi(event, x, y, flags, param):
    global ix, iy, drawing, img_copy, roi
```

**하는 일:**

마우스 이벤트가 발생할 때마다 실행되는 함수를 만듭니다.

**함수의 매개변수:**
- `event` : 마우스 이벤트 종류 (클릭, 이동, 버튼 뗌 등)
- `x, y` : 마우스의 현재 위치 (이미지 내 좌표)
- `flags` : 마우스 버튼 상태
- `param` : 추가 정보 (사용하지 않음)
- `global ...` : 함수 밖의 변수를 사용하겠다는 선언

---

## 3단계: 마우스 좌클릭 - 시작점 지정

```python
if event == cv.EVENT_LBUTTONDOWN:
    drawing = True
    ix, iy = x, y
```

**하는 일:**
- `cv.EVENT_LBUTTONDOWN` : 마우스 좌클릭 이벤트 감지
- `drawing = True` : 드래그 중임을 표시
- `ix, iy = x, y` : 클릭한 위치를 시작 좌표로 저장

마우스를 클릭하면 영역 선택이 시작됩니다.

---

## 4단계: 마우스 이동 - 드래그 중 사각형 표시

```python
elif event == cv.EVENT_MOUSEMOVE:
    if drawing:
        img_copy = img.copy()
        cv.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), 2)
```

**하는 일:**
- `cv.EVENT_MOUSEMOVE` : 마우스가 움직일 때마다 발생
- `if drawing:` : 드래그 중일 때만 실행
- `img_copy = img.copy()` : 원본을 복사 (이전 사각형 잔상 제거)
- `cv.rectangle()` : 사각형을 그림
  - `(ix, iy)` : 시작점
  - `(x, y)` : 현재 마우스 위치 (끝점)
  - `(0, 255, 0)` : 초록색
  - `2` : 선 두께

드래그하면서 실시간으로 선택 영역을 초록색 사각형으로 보여줍니다.

---

## 5단계: 마우스 버튼 뗌 - ROI 추출 및 표시

```python
elif event == cv.EVENT_LBUTTONUP:
    drawing = False
    x1, x2 = min(ix, x), max(ix, x)
    y1, y2 = min(iy, y), max(iy, y)
    
    if x1 != x2 and y1 != y2:
        roi = img[y1:y2, x1:x2]
        cv.imshow('ROI', roi)
```

**하는 일:**
- `cv.EVENT_LBUTTONUP` : 마우스 버튼을 뗄 때 발생
- `drawing = False` : 드래그 종료
- `min(), max()` : 시작점과 끝점의 좌표를 올바르게 정렬
- `roi = img[y1:y2, x1:x2]` : 이미지에서 선택된 영역을 잘라냄
- `cv.imshow('ROI', roi)` : 잘라낸 영역을 별도 창에 표시

마우스 버튼을 떼면 선택된 영역을 추출해서 새로운 창에 보여줍니다.

**참고:** NumPy 슬라이싱 `img[y1:y2, x1:x2]`로 이미지의 일부를 잘라냅니다.

---

## 6단계: 윈도우 생성 및 콜백 등록

```python
cv.namedWindow('Select ROI')
cv.setMouseCallback('Select ROI', draw_roi)
```

**하는 일:**
- `cv.namedWindow()` : 화면에 표시할 창 생성 (이름: 'Select ROI')
- `cv.setMouseCallback()` : 이 창에서 마우스 이벤트가 발생하면 `draw_roi()` 함수 실행

이미지를 띄울 창을 만들고, 마우스 이벤트를 감지하는 기능을 연결합니다.

---

## 7단계: 안내 메시지 출력

```python
print("r: 리셋 / s: 저장 / q: 종료")
```

**하는 일:**

사용자에게 키보드 조작법을 알려줍니다.

---

## 8단계: 메인 루프 시작

```python
while True:
    cv.imshow('Select ROI', img_copy)
    
    key = cv.waitKey(1) & 0xFF
```

**하는 일:**
- `while True:` : 무한 반복 (프로그램 계속 실행)
- `cv.imshow()` : 이미지를 창에 표시
- `cv.waitKey(1)` : 1ms 동안 기다리며 키 입력을 받음
- `& 0xFF` : 키 값을 정정하는 처리

프로그램이 계속 실행되면서 화면을 갱신하고 키 입력을 확인합니다.

---

## 9단계: 키 입력 처리 - 종료

```python
if key == ord('q'):
    break
```

**하는 일:**
- `ord('q')` : 문자 'q'의 ASCII 코드값
- `key == ord('q')` : 'q' 키를 눌렀는지 확인
- `break` : while 루프를 빠져나옴 (프로그램 종료)

'q' 키를 누르면 프로그램이 종료됩니다.

---

## 10단계: 키 입력 처리 - 리셋

```python
elif key == ord('r'):
    img_copy = img.copy()
    if cv.getWindowProperty('ROI', cv.WND_PROP_VISIBLE) >= 1:
        cv.destroyWindow('ROI')
    print("영역 선택이 리셋되었습니다.")
```

**하는 일:**
- `key == ord('r')` : 'r' 키를 눌렀는지 확인
- `img_copy = img.copy()` : 원본 이미지로 복원 (사각형 제거)
- `cv.getWindowProperty()` : 'ROI' 창이 열려있는지 확인
- `cv.destroyWindow('ROI')` : ROI 창이 열려있으면 닫음
- `print()` : 리셋 완료 메시지 출력

'r' 키를 누르면 선택 영역을 초기화하고 ROI 창을 닫습니다.

---

## 11단계: 키 입력 처리 - 저장

```python
elif key == ord('s'):
    if roi is not None:
        cv.imwrite('soccer_roi.jpg', roi)
        print("ROI가 'soccer_roi.jpg'로 저장되었습니다.")
    else:
        print("선택된 영역이 없습니다.")
```

**하는 일:**
- `key == ord('s')` : 's' 키를 눌렀는지 확인
- `if roi is not None:` : ROI가 선택되었는지 확인
- `cv.imwrite()` : 선택된 영역을 파일로 저장
- `print()` : 저장 완료 또는 오류 메시지 출력

's' 키를 누르면 선택된 영역을 'soccer_roi.jpg' 파일로 저장합니다.

---

## 12단계: 모든 창 닫기

```python
cv.destroyAllWindows()
```

**하는 일:**

프로그램이 종료될 때 열려있던 모든 창을 닫습니다.

---

## 전체 흐름 요약

1. 이미지 파일 불러오기
2. 전역 변수 초기화
3. 마우스 이벤트를 처리할 함수 만들기
4. 이미지를 표시할 창 만들기
5. 마우스 이벤트와 함수 연결하기
6. 무한 루프를 실행하면서:
   - 이미지 화면에 표시
   - 마우스 드래그로 영역 선택 및 사각형 표시
   - 마우스 버튼 뗄 때 ROI 추출 및 표시
   - 키 입력으로 리셋, 저장, 종료
7. 모든 창 닫기

---

## 마우스 이벤트 종류

| 이벤트 | 의미 |
|--------|------|
| EVENT_LBUTTONDOWN | 마우스 좌클릭 (누름) |
| EVENT_MOUSEMOVE | 마우스 이동 |
| EVENT_LBUTTONUP | 마우스 좌클릭 (뗌) |

---

## 용어 설명

| 용어 | 의미 |
|------|------|
| ROI (Region of Interest) | 관심 영역, 이미지에서 특정 부분을 선택한 것 |
| 콜백 함수 | 특정 이벤트가 발생하면 자동으로 실행되는 함수 |
| 이벤트 | 마우스 클릭, 키 입력 등 사용자 행동 |
| NumPy 슬라이싱 | 배열에서 특정 범위를 잘라내는 방법 |
| 전역 변수 | 함수 밖에서 선언된 변수로 여러 함수에서 공유 |


---

## 결과
![soccer_full_selection](https://github.com/user-attachments/assets/6009b4b8-8033-4edc-b762-692039460118)

