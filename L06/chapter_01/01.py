"""SORT 알고리즘을 활용한 다중 객체 추적 예제.

이 스크립트는 OpenCV DNN으로 YOLOv3 검출을 수행하고,
검출된 박스를 SORT 방식으로 추적하여 각 객체에 고유 ID를 부여합니다.

실행 예시:
    python 01.py --source 0
    python 01.py --source sample.mp4

모델 파일은 다음 위치에 있어야 합니다.
    L06/yolov3.cfg
    L06/yolov3.weights
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Sequence, Tuple

try:
    import cv2 as cv  # type: ignore[import-not-found]
    import numpy as np
    from scipy.optimize import linear_sum_assignment
except ImportError as exc:  # pragma: no cover - 실행 환경 안내용
    raise SystemExit(
        "필수 패키지가 없습니다. opencv-python, numpy, scipy를 설치하세요."
    ) from exc


def parse_args() -> argparse.Namespace:
    # 실행에 필요한 명령줄 인자를 읽어 옵션 값을 만든다.
    """명령줄 인자를 읽어서 실행 옵션을 정한다."""
    parser = argparse.ArgumentParser(
        description="YOLOv3 + SORT 기반 다중 객체 추적"
    )
    parser.add_argument(
        "--source",
        default="0",
        help="웹캠 번호 또는 비디오 파일 경로(기본값: 0)",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.5,
        help="검출 신뢰도 임계값",
    )
    parser.add_argument(
        "--nms-threshold",
        type=float,
        default=0.4,
        help="NMS 임계값",
    )
    parser.add_argument(
        "--max-age",
        type=int,
        default=20,
        help="검출이 없어도 추적을 유지할 최대 프레임 수",
    )
    parser.add_argument(
        "--min-hits",
        type=int,
        default=3,
        help="안정적인 트랙으로 간주하기 위한 최소 검출 횟수",
    )
    parser.add_argument(
        "--output",
        default="",
        help="결과 영상을 저장할 경로(선택 사항)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="출력 프레임 가로 크기",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="출력 프레임 세로 크기",
    )
    return parser.parse_args()


def resolve_model_paths() -> Tuple[Path, Path]:
    # YOLOv3 설정 파일과 가중치 파일의 경로를 찾는다.
    """YOLO 설정 파일과 가중치 파일의 위치를 찾는다."""
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    cfg_path = project_root / "yolov3.cfg"
    weights_path = project_root / "yolov3.weights"

    if not cfg_path.exists():
        raise FileNotFoundError(f"YOLO 설정 파일을 찾을 수 없습니다: {cfg_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"YOLO 가중치 파일을 찾을 수 없습니다: {weights_path}")

    return cfg_path, weights_path


def open_video_source(source: str) -> cv.VideoCapture:
    # 웹캠 또는 동영상 파일을 열고, 실패하면 샘플 영상으로 대체한다.
    """웹캠 또는 동영상 파일을 연다."""
    script_dir = Path(__file__).resolve().parent
    fallback_video = script_dir.parent / "slow_traffic_small.mp4"

    capture = cv.VideoCapture(int(source)) if source.isdigit() else cv.VideoCapture(source)
    if capture.isOpened():
        return capture

    if source.isdigit() and fallback_video.exists():
        print(
            f"웹캠 {source} 을 열 수 없어 샘플 영상으로 전환합니다: {fallback_video}"
        )
        fallback_capture = cv.VideoCapture(str(fallback_video))
        if fallback_capture.isOpened():
            return fallback_capture

    raise RuntimeError(
        f"비디오 소스를 열 수 없습니다: {source}. 웹캠이 없으면 --source에 비디오 파일 경로를 지정하세요."
    )


def get_output_layer_names(net: cv.dnn_Net) -> List[str]:
    # YOLO 네트워크의 출력 레이어 이름을 가져온다.
    """YOLO 네트워크의 출력 레이어 이름을 가져온다."""
    layer_names = net.getLayerNames()
    unconnected = net.getUnconnectedOutLayers()
    if hasattr(unconnected, "shape") and len(unconnected.shape) == 2:
        indexes = unconnected.flatten()
    else:
        indexes = unconnected
    return [layer_names[index - 1] for index in indexes]


def detect_objects(
    frame: np.ndarray,
    net: cv.dnn_Net,
    output_layer_names: Sequence[str],
    conf_threshold: float,
    nms_threshold: float,
) -> np.ndarray:
    # 프레임에서 객체를 검출하고 NMS로 중복 박스를 정리한다.
    """YOLOv3로 프레임에서 객체를 검출하고 NMS를 적용한다."""
    height, width = frame.shape[:2]
    # DNN 입력용 blob으로 바꾼 뒤 forward를 수행한다.
    blob = cv.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layer_names)

    boxes: List[List[int]] = []
    confidences: List[float] = []

    for output in outputs:
        for detection in output:
            # detection = [center_x, center_y, w, h, objectness, class_scores...]
            scores = detection[5:]
            confidence = float(detection[4] * np.max(scores))

            if confidence < conf_threshold:
                continue

            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            box_width = int(detection[2] * width)
            box_height = int(detection[3] * height)
            x = int(center_x - box_width / 2)
            y = int(center_y - box_height / 2)

            boxes.append([x, y, box_width, box_height])
            confidences.append(confidence)

    if not boxes:
        return np.empty((0, 5), dtype=np.float32)

    # 겹치는 박스는 NMS로 정리해서 중복 검출을 줄인다.
    indices = cv.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    detections: List[List[float]] = []

    if len(indices) > 0:
        for index in np.array(indices).flatten():
            x, y, box_width, box_height = boxes[index]
            x1 = max(0, min(x, width - 1))
            y1 = max(0, min(y, height - 1))
            x2 = max(0, min(x + box_width, width - 1))
            y2 = max(0, min(y + box_height, height - 1))
            if x2 <= x1:
                x2 = min(width - 1, x1 + 1)
            if y2 <= y1:
                y2 = min(height - 1, y1 + 1)
            detections.append([float(x1), float(y1), float(x2), float(y2), float(confidences[index])])

    return np.asarray(detections, dtype=np.float32)


def bbox_to_measurement(bbox: Sequence[float]) -> np.ndarray:
    # 박스 좌표를 칼만필터가 이해하는 측정값 형태로 바꾼다.
    """직사각형 박스를 칼만필터 측정 벡터로 변환한다."""
    x1, y1, x2, y2 = bbox[:4]
    width = x2 - x1
    height = y2 - y1
    center_x = x1 + width / 2.0
    center_y = y1 + height / 2.0
    scale = width * height
    ratio = width / height if height != 0 else 0.0
    return np.array([[center_x], [center_y], [scale], [ratio]], dtype=np.float32)


def measurement_to_bbox(state: np.ndarray) -> np.ndarray:
    # 칼만필터 상태 벡터를 다시 화면에 그릴 박스 좌표로 바꾼다.
    """칼만필터 상태 벡터를 다시 좌표 박스로 되돌린다."""
    center_x, center_y, scale, ratio = state[:4].reshape(-1)
    if scale <= 0 or ratio <= 0:
        return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    width = math.sqrt(scale * ratio)
    height = scale / width if width > 0 else 0.0
    x1 = center_x - width / 2.0
    y1 = center_y - height / 2.0
    x2 = center_x + width / 2.0
    y2 = center_y + height / 2.0
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def iou(bbox_a: Sequence[float], bbox_b: Sequence[float]) -> float:
    # 두 박스가 얼마나 겹치는지 IoU 값으로 계산한다.
    """두 박스의 IoU(겹침 비율)를 계산한다."""
    ax1, ay1, ax2, ay2 = bbox_a[:4]
    bx1, by1, bx2, by2 = bbox_b[:4]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_width = max(0.0, inter_x2 - inter_x1)
    inter_height = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return float(inter_area / union)


def associate_detections_to_trackers(
    detections: np.ndarray,
    trackers: np.ndarray,
    iou_threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    # 검출 결과와 기존 트랙을 IoU 기준으로 매칭한다(헝가리안 알고리즘).
    """검출 결과와 기존 트랙을 IoU 기준으로 매칭한다."""
    if len(trackers) == 0:
        return (
            np.empty((0, 2), dtype=np.int32),
            np.arange(len(detections), dtype=np.int32),
        )

    if len(detections) == 0:
        return (
            np.empty((0, 2), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
        )

    iou_matrix = np.zeros((len(trackers), len(detections)), dtype=np.float32)
    for tracker_index, tracker_bbox in enumerate(trackers):
        for detection_index, detection_bbox in enumerate(detections):
            iou_matrix[tracker_index, detection_index] = iou(tracker_bbox, detection_bbox)

    # linear_sum_assignment는 최소 비용 문제를 푸므로 (1 - IoU)를 비용으로 사용한다.
    cost_matrix = 1.0 - iou_matrix
    tracker_indices, detection_indices = linear_sum_assignment(cost_matrix)

    matched_indices: List[Tuple[int, int]] = []
    used_detections = set()

    for tracker_index, detection_index in zip(tracker_indices, detection_indices):
        if iou_matrix[tracker_index, detection_index] >= iou_threshold:
            matched_indices.append((int(tracker_index), int(detection_index)))
            used_detections.add(int(detection_index))

    unmatched_detections = np.array(
        [index for index in range(len(detections)) if index not in used_detections],
        dtype=np.int32,
    )

    matches = np.asarray(matched_indices, dtype=np.int32) if matched_indices else np.empty((0, 2), dtype=np.int32)

    return matches, unmatched_detections


class KalmanBoxTracker:
    # 단일 객체의 이동을 칼만필터로 추정하는 추적기다.
    """단일 객체의 위치 변화를 칼만필터로 추정한다."""

    count = 0

    def __init__(self, bbox: Sequence[float]) -> None:
        # SORT에서 사용하는 기본 상태 모델.
        self.kf = cv.KalmanFilter(7, 4)
        self.kf.transitionMatrix = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        self.kf.measurementMatrix = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ],
            dtype=np.float32,
        )
        self.kf.processNoiseCov = np.eye(7, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1
        self.kf.errorCovPost = np.eye(7, dtype=np.float32)
        self.kf.statePost = np.zeros((7, 1), dtype=np.float32)
        self.kf.statePost[:4] = bbox_to_measurement(bbox)

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history: List[np.ndarray] = []
        self.hits = 1
        self.hit_streak = 1
        self.age = 0

    def update(self, bbox: Sequence[float]) -> None:
        # 검출 박스를 관측값으로 넣어 필터를 보정한다.
        measurement = bbox_to_measurement(bbox)
        self.kf.correct(measurement)
        self.time_since_update = 0
        self.history.clear()
        self.hits += 1
        self.hit_streak += 1

    def predict(self) -> np.ndarray:
        # 다음 프레임의 위치를 예측한다.
        if float(self.kf.statePost[2, 0] + self.kf.statePost[6, 0]) <= 0:
            self.kf.statePost[6] = 0.0

        prediction = self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        bbox = measurement_to_bbox(prediction)
        self.history.append(bbox)
        return bbox

    def get_state(self) -> np.ndarray:
        return measurement_to_bbox(self.kf.statePost)


class Sort:
    # 여러 객체를 동시에 관리하는 SORT 추적기다.
    """검출 결과를 여러 객체 트랙으로 관리하는 간단한 SORT 구현."""

    def __init__(self, max_age: int = 20, min_hits: int = 3, iou_threshold: float = 0.3) -> None:
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: List[KalmanBoxTracker] = []
        self.frame_count = 0

    def update(self, detections: np.ndarray) -> np.ndarray:
        # 1) 기존 트랙을 먼저 예측한다.
        self.frame_count += 1

        predicted_boxes: List[np.ndarray] = []
        for tracker in self.trackers:
            predicted_boxes.append(tracker.predict())

        if predicted_boxes:
            predicted_array = np.stack(predicted_boxes, axis=0)
        else:
            predicted_array = np.empty((0, 4), dtype=np.float32)

        # 2) 예측 박스와 새 검출 박스를 IoU로 연결한다.
        matches, unmatched_detections = associate_detections_to_trackers(
            detections[:, :4] if len(detections) else np.empty((0, 4), dtype=np.float32),
            predicted_array,
            self.iou_threshold,
        )

        # 3) 매칭된 트랙은 측정값으로 보정한다.
        for tracker_index, detection_index in matches:
            self.trackers[tracker_index].update(detections[detection_index, :4])

        # 4) 새로 등장한 객체는 새 트랙으로 만든다.
        for detection_index in unmatched_detections:
            self.trackers.append(KalmanBoxTracker(detections[detection_index, :4]))

        active_tracks: List[np.ndarray] = []
        trackers_to_remove: List[KalmanBoxTracker] = []

        for tracker in self.trackers:
            # 일정 횟수 이상 안정적으로 관측된 트랙만 화면에 표시한다.
            if tracker.time_since_update < 1 and (
                tracker.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            ):
                bbox = tracker.get_state()
                active_tracks.append(np.concatenate((bbox, np.array([tracker.id + 1], dtype=np.float32))))

            # 너무 오래 갱신되지 않은 트랙은 제거한다.
            if tracker.time_since_update > self.max_age:
                trackers_to_remove.append(tracker)

        for tracker in trackers_to_remove:
            self.trackers.remove(tracker)

        if active_tracks:
            return np.stack(active_tracks, axis=0)
        return np.empty((0, 5), dtype=np.float32)


def draw_tracks(frame: np.ndarray, tracks: np.ndarray) -> None:
    # 추적된 객체의 박스와 ID를 화면에 그린다.
    """트랙 박스와 ID를 화면에 그린다."""
    for track in tracks:
        x1, y1, x2, y2, track_id = track
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        color_seed = int(track_id) * 37
        color = (
            (color_seed * 3) % 255,
            (color_seed * 7) % 255,
            (color_seed * 11) % 255,
        )

        cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv.putText(
            frame,
            f"ID {int(track_id)}",
            (x1, max(20, y1 - 10)),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv.LINE_AA,
        )


def main() -> None:
    # 모델을 불러오고, 영상을 읽으며, 검출과 추적을 반복 실행한다.
    """전체 실행 흐름: 모델 로드 -> 영상 열기 -> 검출/추적 -> 출력."""
    args = parse_args()
    cfg_path, weights_path = resolve_model_paths()

    net = cv.dnn.readNetFromDarknet(str(cfg_path), str(weights_path))
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    output_layer_names = get_output_layer_names(net)

    capture = open_video_source(args.source)
    fps = capture.get(cv.CAP_PROP_FPS)
    writer = None
    if args.output:
        fourcc = cv.VideoWriter_fourcc(*"mp4v")
        writer = cv.VideoWriter(
            args.output,
            fourcc,
            fps if fps > 0 else 30.0,
            (args.width, args.height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"출력 비디오 파일을 만들 수 없습니다: {args.output}")

    tracker = Sort(max_age=args.max_age, min_hits=args.min_hits, iou_threshold=0.3)

    frame_index = 0

    while True:
        success, frame = capture.read()
        if not success:
            break

        # 프레임 크기를 통일해서 추적과 표시가 일관되게 보이도록 한다.
        frame = cv.resize(frame, (args.width, args.height))
        detections = detect_objects(
            frame,
            net,
            output_layer_names,
            args.conf_threshold,
            args.nms_threshold,
        )

        # SORT 추적 결과를 받아서 박스와 ID를 그린다.
        tracks = tracker.update(detections)
        draw_tracks(frame, tracks)

        cv.putText(
            frame,
            f"Frame: {frame_index}",
            (20, 40),
            cv.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv.LINE_AA,
        )
        cv.putText(
            frame,
            f"Detections: {len(detections)}  Tracks: {len(tracks)}",
            (20, 75),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv.LINE_AA,
        )

        # 화면에 결과를 보여준다.
        cv.imshow("YOLOv3 + SORT", frame)

        if writer is not None:
            writer.write(frame)

        key = cv.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break

        frame_index += 1

    capture.release()
    if writer is not None:
        writer.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()