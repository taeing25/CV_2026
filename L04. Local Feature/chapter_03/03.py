import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def resolve_image_path(filename: str) -> Path:
    """입력 파일명을 기준으로 실제 경로를 탐색해 반환합니다."""
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / filename,
        script_dir.parent / filename,
        script_dir.parent.parent / filename,
        Path.cwd() / filename,
        Path.cwd() / "L04. Local Feature" / filename,
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(f"{filename} 파일을 찾을 수 없습니다.")


def pick_second_image() -> str:
    """두 번째 이미지 파일명을 자동 선택합니다(img2/imag2/img3 대응)."""
    names = ["imag2.jpg", "img2.jpg", "img3.jpg"]
    for name in names:
        try:
            resolve_image_path(name)
            return name
        except FileNotFoundError:
            continue

    raise FileNotFoundError("imag2.jpg/img2.jpg/img3.jpg 중 사용할 파일을 찾을 수 없습니다.")


def analyze_homography(H: np.ndarray) -> None:
    """
    호모그래피 행렬을 분석하여 회전각, 스케일, 평행이동을 출력합니다.
    
    호모그래피 H의 구조:
    - 좌상단 2x2: 회전 + 스케일 성분
    - 우측 2x1: tx, ty (평행이동)
    - 하단: 원근 변환 성분
    """
    # 좌상단 2x2 부분을 추출 (회전 + 스케일 정보 포함)
    M = H[:2, :2]
    
    # SVD를 이용해 분해: M = U @ Sigma @ Vt
    U, Sigma, Vt = np.linalg.svd(M)
    R = U @ Vt  # 회전 행렬
    
    # 회전각 추출 (라디안 ← 각도로 변환)
    rotation_angle = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    
    # 스케일 계산 (특이값의 평균)
    scale = np.mean(Sigma)
    
    # 평행이동
    tx, ty = H[0, 2], H[1, 2]
    
    print("\n" + "="*60)
    print("호모그래피(Homography) 분석")
    print("="*60)
    print(f"회전각(Rotation): {rotation_angle:.2f}°")
    print(f"스케일(Scale): {scale:.4f}")
    print(f"평행이동(Translation): tx={tx:.2f}, ty={ty:.2f}")
    print("="*60 + "\n")


def main() -> None:
    # ------------------------------------------------------------
    # [STEP 1] 입력 이미지 로드
    # - 기준 이미지(img1)와 비교 이미지를 로드합니다.
    # - 파일명 오타/버전 차이를 고려해 두 번째 이미지는 자동 선택합니다.
    # ------------------------------------------------------------
    img1_path = resolve_image_path("img1.jpg")
    img2_name = pick_second_image()
    img2_path = resolve_image_path(img2_name)
    img1 = cv.imread(str(img1_path))
    img2 = cv.imread(str(img2_path))

    if img1 is None or img2 is None:
        raise RuntimeError("이미지 로드에 실패했습니다. 경로를 확인하세요.")

    # ------------------------------------------------------------
    # [STEP 2] SIFT 특징 추출
    # - 두 이미지에서 keypoints와 descriptors를 계산합니다.
    # - 이후 매칭/호모그래피 품질은 여기서 얻는 특징 품질의 영향을 크게 받습니다.
    # ------------------------------------------------------------
    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create(nfeatures=1200)
    kp1, desc1 = sift.detectAndCompute(gray1, None)
    kp2, desc2 = sift.detectAndCompute(gray2, None)

    if desc1 is None or desc2 is None:
        raise RuntimeError("특징점 디스크립터를 계산할 수 없습니다.")

    # ------------------------------------------------------------
    # [STEP 3] KNN 매칭 + ratio test
    # - knnMatch(k=2)로 각 점의 후보 2개를 비교합니다.
    # - 1순위 후보가 2순위보다 충분히 가까울 때만 good match로 채택합니다.
    # ------------------------------------------------------------
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)
    knn_matches = bf.knnMatch(desc1, desc2, k=2)
    good_matches = []
    ratio = 0.7
    for pair in knn_matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good_matches.append(m)

    if len(good_matches) < 4:
        raise RuntimeError(
            f"좋은 매칭점이 4개 미만입니다. 현재: {len(good_matches)}"
        )

    # ------------------------------------------------------------
    # [STEP 4] 호모그래피 계산 (RANSAC)
    # - good_matches에서 대응 좌표를 모아 src_pts/dst_pts를 구성합니다.
    # - findHomography(..., RANSAC)로 이상치(outlier)를 견디며 변환행렬 H를 추정합니다.
    # - 계산된 호모그래피에서 회전, 스케일, 평행이동을 분석합니다.
    # ------------------------------------------------------------
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    if H is None:
        raise RuntimeError("호모그래피 계산에 실패했습니다.")
    
    # 호모그래피 분석: 회전각, 스케일, 평행이동 확인
    analyze_homography(H)

    # ------------------------------------------------------------
    # [STEP 5] 이미지 워핑 및 파노라마 정합
    # - 이미지1 코너를 호모그래피로 변환해 전체 결과의 경계를 계산합니다.
    # - 음수 좌표가 생길 수 있으므로 평행이동(translation)을 추가합니다.
    # - 두 이미지를 같은 파노라마 캔버스로 워핑한 뒤 겹치는 영역은 블렌딩합니다.
    # ------------------------------------------------------------
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    corners1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
    corners2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
    warped_corners1 = cv.perspectiveTransform(corners1, H)

    all_corners = np.concatenate((warped_corners1, corners2), axis=0)
    [xmin, ymin] = np.floor(all_corners.min(axis=0).ravel()).astype(int)
    [xmax, ymax] = np.ceil(all_corners.max(axis=0).ravel()).astype(int)

    tx = -xmin if xmin < 0 else 0
    ty = -ymin if ymin < 0 else 0
    pano_w = xmax - xmin
    pano_h = ymax - ymin

    T = np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1],
    ], dtype=np.float64)

    warped_img1 = cv.warpPerspective(img1, T @ H, (pano_w, pano_h))
    warped_img2 = cv.warpPerspective(img2, T, (pano_w, pano_h))

    mask1 = cv.cvtColor(warped_img1, cv.COLOR_BGR2GRAY) > 0
    mask2 = cv.cvtColor(warped_img2, cv.COLOR_BGR2GRAY) > 0
    overlap = mask1 & mask2
    only1 = mask1 & (~mask2)

    aligned = warped_img2.copy()
    aligned[only1] = warped_img1[only1]
    aligned[overlap] = cv.addWeighted(warped_img1[overlap], 0.5, warped_img2[overlap], 0.5, 0)

    # ------------------------------------------------------------
    # [STEP 6] RANSAC inlier 기반 매칭 시각화
    # - mask가 있으면 inlier만 남겨 신뢰도 높은 매칭선을 표시합니다.
    # ------------------------------------------------------------
    inlier_matches = good_matches
    if mask is not None:
        mask_flat = mask.ravel().astype(bool)
        inlier_matches = [m for m, ok in zip(good_matches, mask_flat) if ok]

    max_draw = 80
    matching_result = cv.drawMatches(
        img1,
        kp1,
        img2,
        kp2,
        inlier_matches[:max_draw],
        None,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    # ------------------------------------------------------------
    # [STEP 7] 로그 출력
    # - 특징점/매칭 개수를 출력해 파라미터 튜닝에 활용할 수 있습니다.
    # ------------------------------------------------------------
    print(f"Image 1: {img1_path}")
    print(f"Image 2: {img2_path}")
    print(f"Keypoints img1: {len(kp1)}")
    print(f"Keypoints img2: {len(kp2)}")
    print(f"Good matches (ratio<{ratio}): {len(good_matches)}")
    print(f"Inlier matches (RANSAC): {len(inlier_matches)}")

    # ------------------------------------------------------------
    # [STEP 8] 최종 결과 출력
    # - 왼쪽: 정렬된 이미지(aligned)
    # - 오른쪽: inlier 기반 매칭 결과
    # ------------------------------------------------------------
    plt.figure(figsize=(18, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(cv.cvtColor(aligned, cv.COLOR_BGR2RGB))
    plt.title("Warped Image (Homography Alignment)")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(cv.cvtColor(matching_result, cv.COLOR_BGR2RGB))
    plt.title("Matching Result (SIFT + BF KNN + RANSAC Inliers)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
