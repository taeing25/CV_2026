import cv2
import numpy as np
import glob
import os

# 체크보드 내부 코너 개수
CHECKERBOARD = (9, 6)

# 체크보드 한 칸 실제 크기 (mm)
square_size = 25.0

# 코너 정밀화 조건
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 실제 좌표 생성
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size

# 저장할 좌표
objpoints = [] # 실제 3D 좌표
imgpoints = [] # 이미지 상의 2D 좌표

images = glob.glob("L02.Image_Formation/calibration_images/left*.jpg")

img_size = None

# -----------------------------
# 1. 체크보드 코너 검출
# -----------------------------

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 체크보드 코너 찾기
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    
    # 코너가 검출되면 좌표 저장 (캘리브레이션)
    if ret == True:
        objpoints.append(objp)
        # 코너 정밀화
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        
        # 이미지 크기 저장 (모든 이미지 동일하다고 가정)
        if img_size is None:
            img_size = (img.shape[1], img.shape[0])
        
        # 코너 표시 (옵션)
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

print(f"총 {len(objpoints)} 장의 이미지에서 코너를 검출했습니다.")


# -----------------------------
# 2. 카메라 캘리브레이션
# -----------------------------

# K: 카메라 행렬, dist: 왜곡 계수, rvecs: 회전 벡터, tvecs: 이동 벡터
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

print("Camera Matrix K:")
print(K)

print("\nDistortion Coefficients:")
print(dist)

# -----------------------------
# 3. 왜곡 보정 시각화
# -----------------------------

# 첫 번째 이미지로 왜곡 보정 예시
img = cv2.imread(images[0])
h, w = img.shape[:2]

# 왜곡 보정
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
dst = cv2.undistort(img, K, dist, None, newcameramtx)

# # ROI 적용 (선택사항)
# x, y, w, h = roi
# dst = dst[y:y+h, x:x+w]

combined = cv2.hconcat([img, dst])

    # - 원본과 흑백이 나란히 붙은 사진 저장
cv2.imwrite('calibration.jpg', combined)

# 원본과 보정된 이미지 비교
cv2.imshow('Original', img)
cv2.imshow('Undistorted', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
