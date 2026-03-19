import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# 1. 이미지 로드
img = cv.imread('L03. Edge and Region - Homework/coffee cup.jpg')

if img is None:
    print("이미지를 찾을 수 없습니다.")
else:
    # 2. 초기 마스크 및 배경/전경 모델 초기화 
    # 이미지와 동일한 크기의 마스크 생성
    mask = np.zeros(img.shape[:2], np.uint8)
    
    # 알고리즘 내부에서 사용할 임시 배열 (1, 65) 크기의 float64 형식 
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # 3. 초기 사각형 영역 설정 
    # 실제 이미지 크기에 맞춰 좌표를 조정하세요 (예: x=50, y=50, w=400, h=500)
    rect = (70, 50, 1100, 850) 

    # 4. GrabCut 실행
    # iterCount=5는 5번 반복해서 정밀도를 높인다는 의미입니다.
    cv.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)

    # 5. 마스크 처리 
    # 0(BGD) 또는 2(PR_BGD)인 곳은 0으로, 1(FGD) 또는 3(PR_FGD)인 곳은 1로 설정
    # 즉, 전경인 부분만 남깁니다.
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    
    # 원본 이미지에 마스크를 곱해 배경 제거
    # [:, :, np.newaxis]는 채널 축을 맞춰주기 위함입니다.
    img_result = img * mask2[:, :, np.newaxis]

    # 6. 결과 시각화 
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(mask2, cmap='gray')
    plt.title('GrabCut Mask')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(cv.cvtColor(img_result, cv.COLOR_BGR2RGB))
    plt.title('Result (Object Only)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()