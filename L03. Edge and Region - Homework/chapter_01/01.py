import cv2 as cv                        # 이미지처리, 에지 검출
import numpy as np                      # 수치 계산, 행렬 연산
from matplotlib import pyplot as plt    # 시각화

# 소벨 필터
# 이미지의 미분을 계산하는 필터 - 이미지의 밝기 변화량을 계산하여 에지를 검출하는 데 사용

# 1. 이미지 로드)
img_path = 'L03. Edge and Region - Homework/edgeDetectionImage.jpg'
img = cv.imread(img_path)

if img is None:
    print("이미지를 불러올 수 없습니다. 경로를 확인해주세요.")
else:
    # 2. 그레이스케일 변환 
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # 3. x축 및 y축 방향 에지 검출 
    # dx=1, dy=0 이면 x축 방향 미분 / dx=0, dy=1 이면 y축 방향 미분입니다.
    sobel_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)   #세로선
    sobel_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)   #가로선

    # 4. 에지 강도(Magnitude) 계산 
    # x축과 y축의 변화량을 합쳐서 전체 에지의 세기를 구합니다.
    # 공식: Magnitude = sqrt(sobel_x^2 + sobel_y^2)
    mag = cv.magnitude(sobel_x, sobel_y)

    # 5. 에지 강도 이미지를 uint8로 변환 
    sobel_combined = cv.convertScaleAbs(mag)

    # 6. 결과 시각화 
    plt.figure(figsize=(12, 6))

    # 원본 이미지 출력 (BGR을 RGB로 변환해야 올바른 색상으로 보입니다)
    plt.subplot(1, 2, 1)
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    # 에지 강도 이미지 출력
    plt.subplot(1, 2, 2)
    plt.imshow(sobel_combined, cmap='gray')
    plt.title('Sobel Edge Magnitude')
    plt.axis('off')

    plt.tight_layout()
    plt.show()