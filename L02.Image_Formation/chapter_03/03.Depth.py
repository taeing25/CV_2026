import cv2
import numpy as np
from pathlib import Path

# ============================================================
# 출력 폴더 생성
# ============================================================
output_dir = Path("./outputs")
output_dir.mkdir(parents=True, exist_ok=True)

# ============================================================
# 좌/우 이미지 불러오기
# ============================================================
left_color  = cv2.imread("L02.Image_Formation/left.png")
right_color = cv2.imread("L02.Image_Formation/right.png")

# 이미지 로드 실패 시 즉시 종료 (None이면 파일이 없거나 경로가 틀린 것)
if left_color is None or right_color is None:
    raise FileNotFoundError("좌/우 이미지를 찾지 못했습니다.")


# ============================================================
# 카메라 파라미터 설정
# ============================================================
f = 700.0
B = 0.12

# ============================================================
# ROI (Region of Interest, 관심 영역) 설정
# ============================================================
rois = {
    "Painting": (55,  50,  130, 110),
    "Frog":     (90,  265, 230, 95),
    "Teddy":    (310, 35,  115, 90)
}

# ============================================================
# 그레이스케일 변환
# ============================================================
left_gray  = cv2.cvtColor(left_color,  cv2.COLOR_BGR2GRAY)
right_gray = cv2.cvtColor(right_color, cv2.COLOR_BGR2GRAY)


# ============================================================
# 1. Disparity Map 계산
# ============================================================
stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
disparity_raw = stereo.compute(left_gray, right_gray)

# StereoBM의 결과는 정밀도를 위해 실제값의 16배로 저장됨 (고정소수점)
# 예) 실제 disparity = 12.5 → 저장값 = 200
# → 16으로 나눠서 실제 disparity 값으로 복원
disparity = disparity_raw.astype(np.float32) / 16.0


# ============================================================
# 2. Depth Map 계산 (Z = f × B / d)
# ============================================================
valid_mask = disparity > 0                               # 유효한 픽셀만 True
depth_map  = np.zeros_like(disparity, dtype=np.float32)  # 기본값 0으로 초기화
depth_map[valid_mask] = (f * B) / disparity[valid_mask]  # 유효 픽셀에만 depth 계산


# ============================================================
# 3. ROI별 평균 Disparity / Depth 계산
# ============================================================
results = {}

for name, (x, y, w, h) in rois.items():
    # 이미지 배열에서 ROI 영역만 잘라냄 (numpy 슬라이싱)
    # 주의: 배열 인덱스는 [행(y), 열(x)] 순서
    roi_disp  = disparity[y:y+h, x:x+w]
    roi_depth = depth_map[y:y+h, x:x+w]

    # disparity > 0인 픽셀만 추출 (매칭 실패 픽셀 제거)
    valid_disp_pixels  = roi_disp[roi_disp > 0]
    valid_depth_pixels = roi_depth[roi_depth > 0]

    # 유효 픽셀이 하나도 없는 경우를 대비해 0.0으로 fallback
    avg_disp  = float(np.mean(valid_disp_pixels))  if len(valid_disp_pixels)  > 0 else 0.0
    avg_depth = float(np.mean(valid_depth_pixels)) if len(valid_depth_pixels) > 0 else 0.0

    results[name] = {"avg_disparity": avg_disp, "avg_depth": avg_depth}


# ============================================================
# 4. 결과 출력
#
# 세 ROI의 평균 Disparity와 평균 Depth를 표 형태로 출력하고
# Disparity 기준으로 가장 가깝고 먼 물체를 판별
# ============================================================
print("=" * 50)
print(f"{'ROI':<12} {'Avg Disparity':>15} {'Avg Depth (m)':>15}")
print("-" * 50)
for name, vals in results.items():
    print(f"{name:<12} {vals['avg_disparity']:>15.2f} {vals['avg_depth']:>15.4f}")
print("=" * 50)

# Disparity가 클수록 가까운 물체 → max/min으로 최근접/최원거리 판별
closest  = max(results, key=lambda k: results[k]["avg_disparity"])
farthest = min(results, key=lambda k: results[k]["avg_disparity"])
print(f"\n가장 가까운 ROI : {closest}  (avg disparity = {results[closest]['avg_disparity']:.2f})")
print(f"가장 먼    ROI : {farthest}  (avg disparity = {results[farthest]['avg_disparity']:.2f})")


# ============================================================
# 5. Disparity Map 시각화 (가까울수록 빨강, 멀수록 파랑)
# ============================================================

# 유효하지 않은 픽셀(≤0)을 NaN으로 설정 → 정규화 계산에서 제외
disp_tmp = disparity.copy()
disp_tmp[disp_tmp <= 0] = np.nan

if np.all(np.isnan(disp_tmp)):
    raise ValueError("유효한 disparity 값이 없습니다.")

# 상하위 5% 극단값 제거 후 min/max 결정
d_min = np.nanpercentile(disp_tmp, 5)
d_max = np.nanpercentile(disp_tmp, 95)

# min == max인 경우(모든 값이 같을 때) 0으로 나누기 방지
if d_max <= d_min:
    d_max = d_min + 1e-6

# 0.0 ~ 1.0 범위로 정규화
disp_scaled = (disp_tmp - d_min) / (d_max - d_min)
disp_scaled = np.clip(disp_scaled, 0, 1)  # 범위를 벗어난 값을 0~1로 클리핑

# 정규화된 값을 0~255 uint8로 변환 (NaN 위치는 0 = 검정 유지)
disp_vis = np.zeros_like(disparity, dtype=np.uint8)
valid_disp = ~np.isnan(disp_tmp)                                    # NaN이 아닌 위치
disp_vis[valid_disp] = (disp_scaled[valid_disp] * 255).astype(np.uint8)

# JET 컬러맵 적용: 낮은 값(멀리) → 파랑, 높은 값(가까이) → 빨강
disparity_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)


# ============================================================
# 6. Depth Map 시각화 (가까울수록 빨강, 멀수록 파랑)
# ============================================================
depth_vis = np.zeros_like(depth_map, dtype=np.uint8)

if np.any(valid_mask):
    depth_valid = depth_map[valid_mask]  # 유효한 depth 값만 추출

    # 상하위 5% 제거 후 min/max 결정
    z_min = np.percentile(depth_valid, 5)
    z_max = np.percentile(depth_valid, 95)

    if z_max <= z_min:
        z_max = z_min + 1e-6

    # 0.0 ~ 1.0으로 정규화
    depth_scaled = (depth_map - z_min) / (z_max - z_min)
    depth_scaled = np.clip(depth_scaled, 0, 1)

    # 반전: depth가 작을수록(가까울수록) → 1에 가까운 값 → JET에서 빨강
    depth_scaled = 1.0 - depth_scaled

    # 유효한 픽셀에만 색상값 적용 (나머지는 0 = 검정 유지)
    depth_vis[valid_mask] = (depth_scaled[valid_mask] * 255).astype(np.uint8)

# JET 컬러맵 적용
depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)


# ============================================================
# 7. Left / Right 이미지에 ROI 영역 표시
# ============================================================
left_vis  = left_color.copy()
right_vis = right_color.copy()

for name, (x, y, w, h) in rois.items():
    # 초록색(0, 255, 0) 테두리 사각형, 두께 2픽셀
    cv2.rectangle(left_vis,  (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.rectangle(right_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # ROI 이름을 사각형 위에 표시 (y - 8: 박스 상단에서 8px 위에 글자 배치)
    cv2.putText(left_vis,  name, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(right_vis, name, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


# ============================================================
# 8. 결과 이미지 저장
# ============================================================
cv2.imwrite(str(output_dir / "left_roi.png"),      left_vis)
cv2.imwrite(str(output_dir / "right_roi.png"),     right_vis)
cv2.imwrite(str(output_dir / "disparity_map.png"), disparity_color)
cv2.imwrite(str(output_dir / "depth_map.png"),     depth_color)
print("\n결과 이미지 저장 완료 → outputs/")


# ============================================================
# 9. 4개 이미지를 2×2 격자로 합쳐 저장 및 화면 출력
# ============================================================
top    = np.hstack([left_vis, right_vis])
bottom = np.hstack([disparity_color, depth_color])

# 상단과 하단의 가로 너비가 다를 경우 크기를 맞춰 붙임
# (컬러맵 이미지와 원본 이미지의 크기가 미묘하게 다를 수 있음)
if top.shape[1] != bottom.shape[1]:
    bottom = cv2.resize(bottom, (top.shape[1], top.shape[0]))

combined = np.vstack([top, bottom])
cv2.imwrite(str(output_dir / "combined_result.png"), combined)

# 개별 창으로 결과 표시
cv2.imshow("Left ROI",      left_vis)
cv2.imshow("Right ROI",     right_vis)
cv2.imshow("Disparity Map", disparity_color)
cv2.imshow("Depth Map",     depth_color)
cv2.waitKey(0)           # 0 = 키 입력이 있을 때까지 무한 대기
cv2.destroyAllWindows()  # 열린 모든 OpenCV 창 닫기