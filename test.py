import cv2 as cv
import sys

# 이미지를 읽어 (파일명은 본인이 가진 이미지 이름으로 수정하세요)
img = cv.imread('soccer.jpg') 

if img is None:
    sys.exit('파일이 존재하지 않습니다.')
    
# 이미지 크기를 절반으로 줄임
img = cv.resize(img, dsize=(0,0), fx=0.5, fy=0.5)

# 화면에 출력
cv.imshow('Image Display', img)
print(img[0,0,0], img[0,0,1], img[0,0,2])  # 이미지의 첫 번째 픽셀의 BGR


# 아무 키나 누를 때까지 창을 유지
cv.waitKey()
cv.destroyAllWindows()

print(type(img))  # 이미지 데이터의 타입을 출력
print(img.shape)  # 이미지의 크기와 채널 수를 출력