# 마우스로 그림 그리기 프로그램

마우스로 이미지 위에 그림을 그릴 수 있는 프로그램입니다.

좌클릭으로 파란색, 우클릭으로 빨간색을 그리며, 키를 눌러 붓 크기를 조정할 수 있습니다.

---

## 과제 설명
<img width="694" height="394" alt="image" src="https://github.com/user-attachments/assets/af020d8b-687c-40a4-b7ba-f81b899d601e" />
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

## 1단계: 이미지 로드 및 설정

```python
img = cv.imread('soccer.jpg')

if img is None:
    sys.exit('soccer.jpg 파일을 찾을 수 없습니다.')

brush_size = 5
blue_color = (255, 0, 0)
red_color = (0, 0, 255)
```

**하는 일:**
- `cv.imread('soccer.jpg')` : 같은 폴더에서 soccer.jpg 파일을 불러옴
- `if img is None:` : 파일이 없으면 프로그램 종료
- `brush_size = 5` : 초기 붓 크기를 5로 설정
- `blue_color = (255, 0, 0)` : 파란색 BGR 값
- `red_color = (0, 0, 255)` : 빨간색 BGR 값

그림 그릴 기본 이미지와 색상을 준비합니다.

**참고:** BGR은 파란색, 초록색, 빨간색 순서입니다. (RGB와 반대)

---

## 2단계: 마우스 콜백 함수 정의

```python
def draw(event, x, y, flags, param):
    global brush_size
```

**하는 일:**

마우스 이벤트가 발생할 때마다 실행되는 함수를 만듭니다.

**함수의 매개변수:**
- `event` : 마우스 이벤트 종류 (클릭, 이동 등)
- `x, y` : 마우스의 현재 위치 (이미지 내 좌표)
- `flags` : 마우스 버튼(좌/우/중)이 눌려있는지 상태
- `param` : 추가 정보 (여기서는 사용하지 않음)
- `global brush_size` : 함수 밖의 brush_size 변수를 사용하겠다는 선언

---

## 3단계: 좌클릭 처리

```python
if event == cv.EVENT_LBUTTONDOWN:
    cv.circle(img, (x, y), brush_size, blue_color, -1)
```

**하는 일:**
- `cv.EVENT_LBUTTONDOWN` : 마우스 좌클릭 이벤트 감지
- `cv.circle()` : 원 모양을 그림
  - `img` : 그릴 이미지
  - `(x, y)` : 원의 중심 좌표
  - `brush_size` : 원의 반지름
  - `blue_color` : 파란색
  - `-1` : 원 내부를 채움 (음수이면 채움, 양수이면 테두리만)

마우스로 한 번 클릭하면 파란색 원이 그려집니다.

---

## 4단계: 우클릭 처리

```python
elif event == cv.EVENT_RBUTTONDOWN:
    cv.circle(img, (x, y), brush_size, red_color, -1)
```

**하는 일:**
- `cv.EVENT_RBUTTONDOWN` : 마우스 우클릭 이벤트 감지
- 좌클릭과 동일하지만 빨간색으로 그림

마우스 우클릭으로 빨간색 원이 그려집니다.

---

## 5단계: 드래그 처리

```python
elif event == cv.EVENT_MOUSEMOVE:
    if flags & cv.EVENT_FLAG_LBUTTON:
        cv.circle(img, (x, y), brush_size, blue_color, -1)
    elif flags & cv.EVENT_FLAG_RBUTTON:
        cv.circle(img, (x, y), brush_size, red_color, -1)
```

**하는 일:**
- `cv.EVENT_MOUSEMOVE` : 마우스가 움직일 때마다 발생
- `flags & cv.EVENT_FLAG_LBUTTON` : 좌클릭을 누른 상태로 움직이는지 확인
- `flags & cv.EVENT_FLAG_RBUTTON` : 우클릭을 누른 상태로 움직이는지 확인

마우스를 버튼을 누르면서 드래그하면 계속 원이 그려집니다. 점들이 모여서 선처럼 보입니다.

**참고:** `&` 기호는 상태를 확인하는 연산자이며, "~인지 확인" 정도로 생각하면 됩니다.

---

## 6단계: 윈도우 생성 및 콜백 등록

```python
cv.namedWindow('Painting on Soccer')
cv.setMouseCallback('Painting on Soccer', draw)
```

**하는 일:**
- `cv.namedWindow()` : 화면에 표시할 창 생성 (이름: 'Painting on Soccer')
- `cv.setMouseCallback()` : 이 창에서 마우스 이벤트가 발생하면 `draw()` 함수 실행

이미지를 띄울 창을 만들고, 마우스 이벤트를 감지하는 기능을 연결합니다.

---

## 7단계: 안내 메시지 출력

```python
print("프로그램 시작!")
print("좌클릭: 파란색 / 우클릭: 빨간색")
print("+ 키: 붓 크게 / - 키: 붓 작게 / q 키: 종료")
```

**하는 일:**

사용자에게 프로그램 사용법을 알려줍니다.

---

## 8단계: 메인 루프

```python
while True:
    cv.imshow('Painting on Soccer', img)
    
    key = cv.waitKey(1) & 0xFF
```

**하는 알:**
- `while True:` : 무한 반복 (프로그램이 계속 실행됨)
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
- `key == ord('q')` : 사용자가 'q' 키를 눌렀는지 확인
- `break` : while 루프를 빠져나옴 (프로그램 종료)

사용자가 'q' 키를 누르면 프로그램이 종료됩니다.

---

## 10단계: 키 입력 처리 - 붓 크기 증가

```python
elif key == ord('+') or key == ord('='):
    brush_size = min(15, brush_size + 1)
    print(f"현재 붓 크기: {brush_size}")
```

**하는 일:**
- `ord('+') or ord('=')` : '+' 키 또는 '=' 키를 눌렀는지 확인
- `brush_size + 1` : 붓 크기를 1씩 증가
- `min(15, ...)` : 15를 최대값으로 설정 (15 이상으로는 안 됨)
- `print()` : 현재 붓 크기를 화면에 표시

'+' 또는 '=' 키를 누르면 붓이 커집니다. (최대 15까지)

**참고:** f-string `f"...{변수}..."`를 사용하면 문자열 안에 변수를 넣을 수 있습니다.

---

## 11단계: 키 입력 처리 - 붓 크기 감소

```python
elif key == ord('-'):
    brush_size = max(1, brush_size - 1)
    print(f"현재 붓 크기: {brush_size}")
```

**하는 일:**
- `key == ord('-')` : '-' 키를 눌렀는지 확인
- `brush_size - 1` : 붓 크기를 1씩 감소
- `max(1, ...)` : 1을 최소값으로 설정 (1 이하로는 안 됨)

'-' 키를 누르면 붓이 작아집니다. (최소 1까지)

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
2. 붓 크기와 색상 설정
3. 마우스 이벤트를 처리할 함수 만들기
4. 이미지를 표시할 창 만들기
5. 마우스 이벤트와 함수 연결하기
6. 무한 루프를 실행하면서:
   - 이미지 화면에 표시
   - 마우스 클릭/드래그로 그림 그리기
   - 키 입력으로 붓 크기 조절
   - 'q' 키로 종료
7. 모든 창 닫기

---

## 마우스 이벤트 종류

| 이벤트 | 의미 |
|--------|------|
| EVENT_LBUTTONDOWN | 마우스 좌클릭 |
| EVENT_RBUTTONDOWN | 마우스 우클릭 |
| EVENT_MOUSEMOVE | 마우스 이동 |
| EVENT_FLAG_LBUTTON | 좌클릭 상태 확인 |
| EVENT_FLAG_RBUTTON | 우클릭 상태 확인 |

---

## 용어 설명

| 용어 | 의미 |
|------|------|
| 콜백 함수 | 특정 이벤트가 발생하면 자동으로 실행되는 함수 |
| 이벤트 | 마우스 클릭, 키 입력 등 사용자 행동 |
| 플래그(flag) | 현재 상태를 나타내는 값 |
| BGR | 컴퓨터에서 색상을 표현하는 방식 (파란색, 초록색, 빨간색) |
| 마우스 콜백 | 마우스 이벤트를 감지해서 실행하는 함수 |

---

## 결과
![soccer_painted](https://github.com/user-attachments/assets/cbfdf9b9-355a-4ad5-bb30-7bb47918cadf)
