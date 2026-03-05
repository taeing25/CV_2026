import cv2 as cv
import sys

# 전역 변수 초기화
img = cv.imread('soccer.jpg')  # 이미지 파일 로드
if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

img_copy = img.copy()  # 사각형을 그릴 복사본 생성
roi = None             # 잘라낸 이미지를 담을 변수
ix, iy = -1, -1        # 시작 좌표 초기화
drawing = False        # 마우스 클릭 상태 확인

def draw_roi(event, x, y, flags, param):
    global ix, iy, drawing, img_copy, roi  # 전역 변수 사용 선언

    # 마우스 왼쪽 버튼 클릭: 시작점 지정
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True  # 드래그 시작
        ix, iy = x, y   # 시작 좌표 저장

    # 마우스 이동: 드래그 중인 영역 사각형으로 시각화
    elif event == cv.EVENT_MOUSEMOVE:
        if drawing:
            img_copy = img.copy()  # 이전 사각형 잔상을 지우기 위해 원본 복사
            cv.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), 2)  # 초록색 사각형 그리기

    # 마우스 버튼 뗌: ROI 추출 및 별도 창 출력
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False  # 드래그 종료
        # 시작점과 끝점의 대소 관계를 고려하여 좌표 정렬
        x1, x2 = min(ix, x), max(ix, x)
        y1, y2 = min(iy, y), max(iy, y)

        if x1 != x2 and y1 != y2:  # 유효한 영역인지 확인
            roi = img[y1:y2, x1:x2]  # ROI 추출 (NumPy 슬라이싱)
            cv.imshow('ROI', roi)    # 별도 창에 출력

# 윈도우 생성 및 마우스 콜백 등록
cv.namedWindow('Select ROI')  # 이미지 표시 창 생성
cv.setMouseCallback('Select ROI', draw_roi)  # 마우스 이벤트 콜백 함수 연결

print("r: 리셋 / s: 저장 / q: 종료")  # 사용법 안내

while True:
    cv.imshow('Select ROI', img_copy)  # 이미지 창에 표시

    key = cv.waitKey(1) & 0xFF  # 키 입력 대기 (1ms)

    # q 키를 누르면 종료
    if key == ord('q'):
        break

    # r 키를 누르면 영역 선택 리셋
    elif key == ord('r'):
        img_copy = img.copy()  # 원본으로 복원
        if cv.getWindowProperty('ROI', cv.WND_PROP_VISIBLE) >= 1:
            cv.destroyWindow('ROI')  # ROI 창 닫기
        print("영역 선택이 리셋되었습니다.")

    # s 키를 누르면 선택한 영역을 파일로 저장
    elif key == ord('s'):
        if roi is not None:
            cv.imwrite('soccer_roi.jpg', roi)  # 이미지 파일로 저장
            cv.imwrite('soccer_full_selection.jpg', img)  # 전체 선택 영역이 포함된 이미지도 저장
            print("ROI가 'soccer_roi.jpg'로 저장되었습니다.")
        else:
            print("선택된 영역이 없습니다.")

cv.destroyAllWindows()  # 모든 창 닫기