import cv2 as cv
import matplotlib.pyplot as plt
from pathlib import Path


def resolve_image_path(filename: str) -> Path:
    """이미지 파일 경로를 여러 후보 위치에서 탐색해 반환합니다.

    실행 위치가 달라도 동일 코드가 동작하도록 경로 후보를 순차적으로 확인합니다.
    """
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


def match_with_bf(desc1, desc2):
    """BFMatcher(crossCheck=True)로 단순하고 직관적인 1:1 매칭을 수행합니다."""
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key=lambda m: m.distance)
    return matches


def match_with_flann_ratio_test(desc1, desc2, ratio=0.75):
    """FLANN + KNN + ratio test 방식으로 신뢰도 높은 매칭만 선별합니다."""
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)

    knn_matches = flann.knnMatch(desc1, desc2, k=2)
    good_matches = []

    # knnMatch는 각 특징점에 대해 최근접 이웃 2개를 반환합니다.
    # m(1순위)와 n(2순위)의 거리 차이가 충분히 클 때만 매칭을 채택하면,
    # 애매한 대응점을 줄여 매칭 품질을 높일 수 있습니다.
    for pair in knn_matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good_matches.append(m)

    good_matches = sorted(good_matches, key=lambda m: m.distance)
    return good_matches


def main() -> None:
    # ------------------------------------------------------------
    # [STEP 1] 입력 이미지 로드
    # ------------------------------------------------------------
    img1_path = resolve_image_path("mot_color70.jpg")
    img2_path = resolve_image_path("mot_color83.jpg")
    img1 = cv.imread(str(img1_path))
    img2 = cv.imread(str(img2_path))

    if img1 is None or img2 is None:
        raise RuntimeError("이미지 로드에 실패했습니다. 파일 경로를 확인하세요.")

    # ------------------------------------------------------------
    # [STEP 2] SIFT 특징 추출
    # - 두 이미지 각각에서 keypoints / descriptors를 계산합니다.
    # - descriptor는 이후 매칭 단계에서 거리 비교의 기준이 됩니다.
    # ------------------------------------------------------------
    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create(nfeatures=500)
    kp1, desc1 = sift.detectAndCompute(gray1, None)
    kp2, desc2 = sift.detectAndCompute(gray2, None)

    if desc1 is None or desc2 is None:
        raise RuntimeError("특징점 디스크립터를 계산하지 못했습니다.")

    # ------------------------------------------------------------
    # [STEP 3] 특징점 매칭
    # - matcher_type을 "BF" 또는 "FLANN"으로 바꿔 방식 비교가 가능합니다.
    # ------------------------------------------------------------
    matcher_type = "BF"
    if matcher_type == "FLANN":
        matches = match_with_flann_ratio_test(desc1, desc2, ratio=0.75)
    else:
        matches = match_with_bf(desc1, desc2)

    # ------------------------------------------------------------
    # [STEP 4] 시각화용 매칭 수 제한
    # - 모든 매칭을 그리면 복잡해질 수 있어 상위 일부만 표시합니다.
    # ------------------------------------------------------------
    max_draw = 80
    matches_to_draw = matches[:max_draw]

    # ------------------------------------------------------------
    # [STEP 5] 매칭 결과 시각화
    # - drawMatches는 두 이미지를 좌우로 배치하고 대응점을 선으로 연결합니다.
    # ------------------------------------------------------------
    matched_img = cv.drawMatches(
        img1,
        kp1,
        img2,
        kp2,
        matches_to_draw,
        None,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    # ------------------------------------------------------------
    # [STEP 6] 로그 출력
    # ------------------------------------------------------------
    print(f"Image 1: {img1_path}")
    print(f"Image 2: {img2_path}")
    print(f"Keypoints in image1: {len(kp1)}")
    print(f"Keypoints in image2: {len(kp2)}")
    print(f"Total matches: {len(matches)}")
    print(f"Drawn matches: {len(matches_to_draw)}")

    # ------------------------------------------------------------
    # [STEP 7] matplotlib 출력
    # - OpenCV는 BGR, matplotlib는 RGB이므로 변환 후 출력합니다.
    # ------------------------------------------------------------
    plt.figure(figsize=(16, 7))
    plt.imshow(cv.cvtColor(matched_img, cv.COLOR_BGR2RGB))
    plt.title(f"SIFT Feature Matching ({matcher_type})")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
