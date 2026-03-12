import cv2
import numpy as np

# 이미지 읽기
img = cv2.imread('L02.image_Formation/rose.png') # 실습에 사용할 이미지
if img is None:
    print("이미지를 찾을 수 없습니다")
    exit()

h, w = img.shape[:2]
center = (w / 2, h / 2)

# 1. 회전(+30도) 및 크기 조절(0.8) 행렬 생성
# 이미지 중심 기준으로 30도 회전하면서 0.8배로 줄임
M = cv2.getRotationMatrix2D(center, 30, 0.8)

# 2. 평행이동 반영 (x축 +80px, y축 -40px)
# 회전 행렬 M의 마지막 열이 이동(Translation)을 담당해
M[0, 2] += 80
M[1, 2] -= 40

# 3. 아핀 변환 적용
dst = cv2.warpAffine(img, M, (w, h))

# 결과 출력
cv2.imshow('Original', img)
cv2.imshow('Transformed', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

combined = cv2.hconcat([img, dst])

    # - 원본과 흑백이 나란히 붙은 사진 저장
cv2.imwrite('rose.jpg', combined)