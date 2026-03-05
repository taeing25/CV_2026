import cv2 as cv
import sys

# 이미지를 읽어옵니다. (파일명은 본인이 가진 이미지 이름으로 수정하세요)
img = cv.imread('soccer.jpg') 

if img is None:
    sys.exit('파일이 존재하지 않습니다.')

# 화면에 출력합니다.
cv.imshow('Image Display', img)

# 아무 키나 누를 때까지 창을 유지합니다.
cv.waitKey()
cv.destroyAllWindows()