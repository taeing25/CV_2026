import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# 캐니 에지: 이미지의 밝기 변화량이 급격한 부분을 검출하는 알고리즘
    # 노이즈 제거 후 에지 두께 얇게 검출
# 허프 변환: 이미지에서 직선, 원 등 특정 형태를 검출하는 알고리즘
    # 캐니 에지로 검출된 에지 맵에서 직선을 검출

# 1. 이미지 로드 
img = cv.imread('L03. Edge and Region - Homework/dabo.jpg') 

if img is None:
    print("이미지를 찾을 수 없습니다.")
else:
    # 2. 그레이스케일 변환 
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # 3. 캐니 에지 검출 (더 엄격한 기준으로 명확한 에지만 추출)
    # edges 변수에는 0(검정)과 255(흰색)로 이루어진 이진 에지 맵이 저장됩니다.
    edges = cv.Canny(gray, 150, 250)

    # 4. 확률적 허프 변환을 이용한 직선 검출 
    # 이미지 특성에 따라 이 값들을 조정(튜닝)하는 것
    # edges: 캐니 에지 검출 결과 (이진 이미지, 0과 255로만 구성)
    # rho: 허프 공간에서 거리 해상도 (픽셀 단위, 1 = 1픽셀 간격으로 투표)
    # theta: 허프 공간에서 각도 해상도 (라디안 단위, np.pi/180 = 1도 간격)
    # threshold: 직선으로 인정하는 최소 투표 개수 (70 = 강한 직선 중심)
    # minLineLength: 검출할 최소 직선 길이 (픽셀, 100 = 주요 구조선만)
    # maxLineGap: 직선으로 연결할 수 있는 최대 간격 (픽셀, 8 = 적당히 연결)
    lines = cv.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=70,
                       minLineLength=100, maxLineGap=8)

    # 5. 검출된 직선 그리기
    line_img = img.copy() # 원본 보존을 위해 복사본 생성
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # 색상 (0, 0, 255)는 BGR 기준 빨간색, 두께는 2 
            cv.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # 6. 결과 시각화
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB)) # BGR을 RGB로!
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv.cvtColor(line_img, cv.COLOR_BGR2RGB))
    plt.title('Detected Lines (Red)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()