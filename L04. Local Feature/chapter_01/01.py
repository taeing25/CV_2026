import cv2 as cv
import matplotlib.pyplot as plt
from pathlib import Path


def resolve_image_path() -> Path:
    """mot_color70.jpg 파일 경로를 유연하게 탐색해 반환합니다.

    실습 파일을 어디서 실행하든 동작하도록 여러 후보 경로를 순차적으로 확인합니다.
    """
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / "mot_color70.jpg",
        script_dir.parent / "mot_color70.jpg",
        script_dir.parent.parent / "mot_color70.jpg",
        Path.cwd() / "mot_color70.jpg",
        Path.cwd() / "L04. Local Feature" / "mot_color70.jpg",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "mot_color70.jpg 파일을 찾을 수 없습니다. "
        "chapter_01 폴더 또는 L04. Local Feature 폴더에 이미지를 두고 다시 실행하세요."
    )


def main() -> None:
    # ------------------------------------------------------------
    # [STEP 1] 입력 이미지 로드
    # - resolve_image_path()로 실제 파일 위치를 먼저 찾고,
    # - cv.imread()로 BGR 컬러 이미지를 메모리에 올립니다.
    # ------------------------------------------------------------
    image_path = resolve_image_path()
    img = cv.imread(str(image_path))

    if img is None:
        raise RuntimeError(f"이미지를 읽지 못했습니다: {image_path}")

    # ------------------------------------------------------------
    # [STEP 2] SIFT 특징점 검출 및 디스크립터 계산
    # - SIFT는 grayscale 입력을 사용하므로 먼저 흑백 변환합니다.
    # - nfeatures는 추출할 특징점 최대 개수입니다.
    #   (값이 크면 더 많이 검출, 작으면 핵심점 위주로 검출)
    # ------------------------------------------------------------
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create(nfeatures=500)
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    # ------------------------------------------------------------
    # [STEP 3] 특징점 시각화
    # - DRAW_RICH_KEYPOINTS 플래그를 사용하면
    #   키포인트의 '위치'뿐 아니라 '크기/방향'도 함께 확인할 수 있습니다.
    # ------------------------------------------------------------
    img_keypoints = cv.drawKeypoints(
        img,
        keypoints,
        None,
        flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )

    # ------------------------------------------------------------
    # [STEP 4] 콘솔 로그 출력
    # - 실제로 몇 개 특징점이 검출됐는지,
    # - 디스크립터 행렬 크기는 어떻게 되는지 확인합니다.
    # ------------------------------------------------------------
    print(f"Image: {image_path}")
    print(f"Detected keypoints: {len(keypoints)}")
    if descriptors is not None:
        print(f"Descriptor shape: {descriptors.shape}")

    # ------------------------------------------------------------
    # [STEP 5] 결과 시각화 (원본 vs 특징점 결과)
    # - matplotlib는 RGB를 사용하므로 BGR -> RGB 변환이 필요합니다.
    # ------------------------------------------------------------
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(cv.cvtColor(img_keypoints, cv.COLOR_BGR2RGB))
    plt.title("SIFT Keypoints (RICH)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
