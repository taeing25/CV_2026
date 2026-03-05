import cv2 as cv
import numpy as np
import sys

# 1. cv.imread()를 사용하여 이미지 로드
# 'soccer.jpg' 파일이 파이썬 파일과 같은 폴더에 있어야 합니다.
img = cv.imread('soccer.jpg') 

# 이미지가 제대로 로드되지 않았을 경우 처리
if img is None:
    sys.exit('파일을 찾을 수 없습니다. 파일명과 경로를 확인하세요.')

# 이미지 크기를 절반으로 줄임
img = cv.resize(img, dsize=(0,0), fx=0.5, fy=0.5)

# 2. cv.cvtColor() 함수를 사용해 이미지를 그레이스케일로 변환
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 원본 이미지(img)는 3채널(BGR)이고, 변환된 gray는 1채널
# np.hstack()으로 두 이미지를 붙이려면 채널 수가 같아야 하므로,
# gray 이미지를 다시 3채널 형식으로 변환 (색상은 여전히 회색).
gray_3channel = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)

# 3. np.hstack() 함수를 이용해 원본 이미지와 그레이스케일 이미지를 가로로 연결
combined = np.hstack((img, gray_3channel))

    # - 원본과 흑백이 나란히 붙은 사진 저장
cv.imwrite('soccer_combined.jpg', combined)

# 4. cv.imshow()와 cv.waitKey()를 사용해 결과를 화면에 표시
cv.imshow('Original vs Grayscale', combined)

# 아무 키나 누르면 창이 닫히도록 설정
cv.waitKey(0)
cv.destroyAllWindows()