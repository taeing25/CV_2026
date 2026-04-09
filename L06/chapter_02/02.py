import os

# TensorFlow/absl 초기 로그를 줄이기 위한 환경변수 (mediapipe import 전에 설정)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("GLOG_minloglevel", "2")

import cv2
import mediapipe as mp
import numpy as np
from urllib.request import urlretrieve


def load_image_unicode_safe(path: str):
    image = cv2.imread(path)
    if image is not None:
        return image

    # Windows에서 한글 경로를 cv2.imread가 읽지 못하는 경우를 대비한 우회 로딩
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def get_face_landmarker():
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
        return {
            "mode": "solutions",
            "face_mesh": mp.solutions.face_mesh,
            "drawing": mp.solutions.drawing_utils,
        }

    task_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")
    if not os.path.exists(task_model_path):
        print("mediapipe.tasks 모델을 다운로드합니다... (최초 1회)")
        urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
            task_model_path,
        )

    vision = mp.tasks.vision
    options = vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=task_model_path),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)
    return {
        "mode": "tasks",
        "landmarker": landmarker,
    }


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(base_dir, "수지.png")

    image = load_image_unicode_safe(image_path)
    if image is None:
        print(f"이미지를 불러올 수 없습니다: {image_path}")
        print("파일 경로에 한글이 포함된 경우 OpenCV 빌드에 따라 실패할 수 있습니다.")
        return

    detector = get_face_landmarker()
    output = image.copy()
    h, w, _ = output.shape

    if detector["mode"] == "solutions":
        mp_face_mesh = detector["face_mesh"]
        mp_drawing = detector["drawing"]

        # refine_landmarks=False: 기본 FaceMesh 468 포인트
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
        ) as face_mesh:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # 연결선 시각화
                    mp_drawing.draw_landmarks(
                        image=output,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing.DrawingSpec(
                            color=(0, 255, 0), thickness=1, circle_radius=1
                        ),
                    )

                    # 정규화 좌표를 픽셀 좌표로 변환해 점 표시
                    for lm in face_landmarks.landmark:
                        x = int(lm.x * w)
                        y = int(lm.y * h)
                        cv2.circle(output, (x, y), 1, (0, 0, 255), -1)

                cv2.putText(
                    output,
                    "FaceMesh landmarks detected",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
            else:
                cv2.putText(
                    output,
                    "No face landmarks detected",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector["landmarker"].detect(mp_image)

        if result.face_landmarks:
            for face_landmarks in result.face_landmarks:
                for lm in face_landmarks:
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    if 0 <= x < w and 0 <= y < h:
                        cv2.circle(output, (x, y), 1, (0, 0, 255), -1)

            cv2.putText(
                output,
                f"Face landmarks detected ({len(result.face_landmarks[0])} points)",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                output,
                "No face landmarks detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        detector["landmarker"].close()

    print("ESC 키를 누르면 종료됩니다.")

    while True:
        cv2.imshow("FaceMesh (Image Mode)", output)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
