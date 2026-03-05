import cv2 as cv
import sys

# 이미지 로드
img = cv.imread('soccer.jpg')  # soccer.jpg 파일 불러오기

if img is None:
    sys.exit('soccer.jpg 파일을 찾을 수 없습니다.')  # 파일 없으면 프로그램 종료

# 초기 설정
brush_size = 5               # 초기 붓 크기 설정
blue_color = (255, 0, 0)     # 좌클릭: 파란색 (BGR 형식)
red_color = (0, 0, 255)      # 우클릭: 빨간색 (BGR 형식)

# 마우스 콜백 함수 정의
def draw(event, x, y, flags, param):
    global brush_size  # 전역 변수 사용 선언

    # 좌클릭: 파란색 원 그리기
    if event == cv.EVENT_LBUTTONDOWN:
        cv.circle(img, (x, y), brush_size, blue_color, -1)  # 클릭 위치에 파란색 원 그리기

    # 우클릭: 빨간색 원 그리기
    elif event == cv.EVENT_RBUTTONDOWN:
        cv.circle(img, (x, y), brush_size, red_color, -1)   # 클릭 위치에 빨간색 원 그리기

    # 마우스 이동 (드래그 확인)
    elif event == cv.EVENT_MOUSEMOVE:
        if flags & cv.EVENT_FLAG_LBUTTON:  # 좌클릭 상태로 드래그 중
            cv.circle(img, (x, y), brush_size, blue_color, -1)  # 파란색으로 선 그리기
        elif flags & cv.EVENT_FLAG_RBUTTON:  # 우클릭 상태로 드래그 중
            cv.circle(img, (x, y), brush_size, red_color, -1)   # 빨간색으로 선 그리기

# 윈도우 생성 및 콜백 함수 등록
cv.namedWindow('Painting on Soccer')  # 이미지 표시 창 생성
cv.setMouseCallback('Painting on Soccer', draw)  # 마우스 이벤트 콜백 함수 연결

print("프로그램 시작!")
print("좌클릭: 파란색 / 우클릭: 빨간색")
print("+ 키: 붓 크게 / - 키: 붓 작게 / q 키: 종료")

while True:
    cv.imshow('Painting on Soccer', img)  # 이미지 창에 표시

    # 1ms 대기하며 키 입력 확인
    key = cv.waitKey(1) & 0xFF

    # q 누르면 종료
    if key == ord('q'):
        # 변경된 이미지를 저장
        cv.imwrite('soccer_painted.jpg', img)  # 그린 이미지를 파일로 저장
        break

    # + 또는 = 누르면 붓 크기 증가 (최대 15)
    elif key == ord('+') or key == ord('='):
        brush_size = min(15, brush_size + 1)  # 붓 크기 1 증가 (최대 15)
        print(f"현재 붓 크기: {brush_size}")

    # - 누르면 붓 크기 감소 (최소 1)
    elif key == ord('-'):
        brush_size = max(1, brush_size - 1)  # 붓 크기 1 감소 (최소 1)
        print(f"현재 붓 크기: {brush_size}")

cv.destroyAllWindows()  # 모든 창 닫기