import os

# ============================================================================
# 환경 변수 설정 (TensorFlow/MediaPipe 로그 최소화)
# ============================================================================
# mediapipe를 import하기 전에 설정해야 효과가 있음
# TF_CPP_MIN_LOG_LEVEL: TensorFlow 로그 레벨 (0=모두, 1=정보만, 2=경고만, 3=에러만)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
# TF_ENABLE_ONEDNN_OPTS: oneDNN 최적화 비활성화 (불필요한 경고 제거)
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
# GLOG_minloglevel: Google Logging 최소 로그 레벨 설정
os.environ.setdefault("GLOG_minloglevel", "2")

# ============================================================================
# 필수 라이브러리 import
# ============================================================================
import cv2  # 컴퓨터 비전 작업 (이미지 로드, 시각화)
import mediapipe as mp  # Google의 MediaPipe (얼굴 랜드마크 감지)
import numpy as np  # 수치 계산 및 배열 처리
from urllib.request import urlretrieve  # URL에서 파일 다운로드


def load_image_unicode_safe(path: str):
    """
    유니코드(한글 등) 경로가 포함된 이미지 파일을 안전하게 로드하는 함수
    
    Windows에서 cv2.imread()는 한글 경로를 읽지 못하는 경우가 있으므로,
    우회 방법(파일을 바이너리로 읽어 디코딩)을 제공합니다.
    
    Parameters:
        path (str): 로드할 이미지 파일의 경로
        
    Returns:
        numpy.ndarray: BGR 형식의 이미지 배열, 실패 시 None
    """
    # 방법 1: 표준 cv2.imread() 시도
    image = cv2.imread(path)
    if image is not None:
        return image

    # 방법 2: Windows 한글 경로 대응 - 파일을 바이너리로 읽어 디코딩
    try:
        # 파일을 uint8 바이너리 배열로 읽음
        data = np.fromfile(path, dtype=np.uint8)
        # 파일이 제대로 읽어졌는지 확인
        if data.size == 0:
            return None
        # 바이너리 데이터를 이미지로 디코딩
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        # 예외 발생 시 None 반환
        return None


def get_face_landmarker():
    """
    얼굴 랜드마크 감지 모델을 초기화하는 함수
    
    MediaPipe의 두 가지 API를 지원합니다:
    1. mp.solutions.face_mesh (구 버전, 468개 포인트)
    2. mp.tasks.vision.FaceLandmarker (신 버전, 더 정확함)
    
    Returns:
        dict: 감지기 정보 및 모자이크 객체를 포함한 딕셔너리
    """
    # ===== 방법 1: MediaPipe Solutions 사용 (구 API) =====
    # 최신 mediapipe가 아닌 경우 이 방법 사용
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
        return {
            "mode": "solutions",  # 사용한 API 모드 표시
            "face_mesh": mp.solutions.face_mesh,  # 얼굴 메시 모델
            "drawing": mp.solutions.drawing_utils,  # 그리기 유틸리티
        }

    # ===== 방법 2: MediaPipe Tasks 사용 (신 API) =====
    # 모델 파일 경로 설정 (현재 스크립트 디렉토리에 저장)
    task_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")
    
    # 모델 파일이 없으면 다운로드
    if not os.path.exists(task_model_path):
        print("mediapipe.tasks 모델을 다운로드합니다... (최초 1회)")
        urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
            task_model_path,
        )

    # MediaPipe Vision 모듈에서 얼굴 랜드마커 옵션 설정
    vision = mp.tasks.vision
    options = vision.FaceLandmarkerOptions(
        # 모델 파일 경로
        base_options=mp.tasks.BaseOptions(model_asset_path=task_model_path),
        # 정적 이미지 모드 사용 (비디오 아님)
        running_mode=vision.RunningMode.IMAGE,
        # 감지할 최대 얼굴 수
        num_faces=1,
    )
    
    # 랜드마커 객체 생성
    landmarker = vision.FaceLandmarker.create_from_options(options)
    return {
        "mode": "tasks",  # 사용한 API 모드 표시
        "landmarker": landmarker,  # 랜드마커 객체
    }


def main() -> None:
    """
    메인 함수: 얼굴 랜드마크 감지 및 시각화
    
    프로세스:
    1. 이미지 로드
    2. 얼굴 랜드마크 감지기 초기화
    3. 얼굴 랜드마크 감지 및 시각화
    4. 결과를 화면에 표시
    """
    # ===== 1단계: 이미지 로드 =====
    # 현재 스크립트의 디렉토리 경로
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # 처리할 이미지 파일 경로 (같은 폴더의 "수지.png")
    image_path = os.path.join(base_dir, "수지.png")

    # 유니코드 안전 함수로 이미지 로드
    image = load_image_unicode_safe(image_path)
    if image is None:
        print(f"이미지를 불러올 수 없습니다: {image_path}")
        print("파일 경로에 한글이 포함된 경우 OpenCV 빌드에 따라 실패할 수 있습니다.")
        return

    # ===== 2단계: 얼굴 랜드마크 감지기 초기화 =====
    detector = get_face_landmarker()
    # 원본 이미지를 복사하여 결과 시각화용 이미지 생성
    output = image.copy()
    # 이미지의 높이, 너비, 채널 수 추출
    h, w, _ = output.shape

    # ===== 3단계: 얼굴 랜드마크 감지 및 시각화 =====
    if detector["mode"] == "solutions":
        # ----- API 모드 1: MediaPipe Solutions (구 버전) 사용 -----
        mp_face_mesh = detector["face_mesh"]
        mp_drawing = detector["drawing"]

        # 얼굴 메시 모델 초기화
        # static_image_mode: 비디오가 아닌 정적 이미지 처리 모드
        # max_num_faces: 감지할 최대 얼굴 수
        # refine_landmarks: False = 기본 468개 포인트만 감지 (True면 478개)
        # min_detection_confidence: 신뢰도 임계값 (50% 이상만 인식)
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
        ) as face_mesh:
            # BGR에서 RGB로 변환 (MediaPipe는 RGB 포맷 요구)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # 이미지에서 얼굴 랜드마크 감지
            results = face_mesh.process(rgb)

            # 얼굴이 감지된 경우
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # --- 연결선 시각화: 랜드마크 포인트들을 선으로 연결 ---
                    mp_drawing.draw_landmarks(
                        image=output,  # 그릴 이미지
                        landmark_list=face_landmarks,  # 랜드마크 포인트 목록
                        connections=mp_face_mesh.FACEMESH_TESSELATION,  # 연결 구조
                        landmark_drawing_spec=None,  # 포인트 스타일 (None=기본값)
                        connection_drawing_spec=mp_drawing.DrawingSpec(
                            color=(0, 255, 0),  # 초록색
                            thickness=1,  # 선 두께
                            circle_radius=1  # 원 반지름
                        ),
                    )

                    # --- 개별 랜드마크 포인트를 빨간 점으로 표시 ---
                    for lm in face_landmarks.landmark:
                        # 정규화 좌표 (0~1)를 픽셀 좌표로 변환
                        x = int(lm.x * w)
                        y = int(lm.y * h)
                        # 빨간색 원 그리기
                        cv2.circle(output, (x, y), 1, (0, 0, 255), -1)

                # 성공 메시지 출력
                cv2.putText(
                    output,
                    "FaceMesh landmarks detected",
                    (10, 30),  # 텍스트 위치 (좌상단)
                    cv2.FONT_HERSHEY_SIMPLEX,  # 글꼴
                    0.8,  # 글자 크기
                    (255, 255, 0),  # 색상 (시안색)
                    2,  # 텍스트 두께
                    cv2.LINE_AA,  # 안티앨리어싱
                )
            else:
                # 얼굴이 감지되지 않은 경우
                cv2.putText(
                    output,
                    "No face landmarks detected",
                    (10, 30),  # 텍스트 위치
                    cv2.FONT_HERSHEY_SIMPLEX,  # 글꼴
                    0.8,  # 글자 크기
                    (0, 0, 255),  # 색상 (빨강색)
                    2,  # 텍스트 두께
                    cv2.LINE_AA,  # 안티앨리어싱
                )
    else:
        # ----- API 모드 2: MediaPipe Tasks (신 버전) 사용 -----
        # BGR에서 RGB로 변환 (MediaPipe 요구사항)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # MediaPipe용 이미지 객체 생성
        # image_format=SRGB: RGB 색상 공간
        # data: RGB 이미지 배열
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        # 얼굴 랜드마크 감지 실행
        result = detector["landmarker"].detect(mp_image)

        # 얼굴이 감지된 경우
        if result.face_landmarks:
            # 감지된 각 얼굴의 랜드마크 포인트 시각화
            for face_landmarks in result.face_landmarks:
                for lm in face_landmarks:
                    # 정규화 좌표를 픽셀 좌표로 변환
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    
                    # 이미지 범위 내에 있는지 확인
                    if 0 <= x < w and 0 <= y < h:
                        # 빨간 포인트 그리기
                        cv2.circle(output, (x, y), 1, (0, 0, 255), -1)

            # 감지된 포인트 개수와 함께 성공 메시지 출력
            cv2.putText(
                output,
                f"Face landmarks detected ({len(result.face_landmarks[0])} points)",
                (10, 30),  # 텍스트 위치
                cv2.FONT_HERSHEY_SIMPLEX,  # 글꼴
                0.7,  # 글자 크기
                (255, 255, 0),  # 색상 (시안색)
                2,  # 텍스트 두께
                cv2.LINE_AA,  # 안티앨리어싱
            )
        else:
            # 얼굴이 감지되지 않은 경우
            cv2.putText(
                output,
                "No face landmarks detected",
                (10, 30),  # 텍스트 위치
                cv2.FONT_HERSHEY_SIMPLEX,  # 글꼴
                0.8,  # 글자 크기
                (0, 0, 255),  # 색상 (빨강색)
                2,  # 텍스트 두께
                cv2.LINE_AA,  # 안티앨리어싱
            )

        # 랜드마커 리소스 해제
        detector["landmarker"].close()

    # ===== 4단계: 결과 표시 =====
    print("ESC 키를 누르면 종료됩니다.")

    # 윈도우에 결과 이미지 표시 및 키 입력 대기
    while True:
        # 결과 이미지를 "FaceMesh (Image Mode)" 윈도우에 표시
        cv2.imshow("FaceMesh (Image Mode)", output)
        
        # 1ms 대기 후 키 입력 확인 (1ms마다 갱신)
        key = cv2.waitKey(1) & 0xFF
        
        # ESC 키(ASCII 27)가 눌리면 루프 종료
        if key == 27:  # ESC
            break

    # 열린 모든 cv2 윈도우 종료
    cv2.destroyAllWindows()


# ============================================================================
# 프로그램 시작점
# ============================================================================
if __name__ == "__main__":
    # 이 스크립트가 직접 실행될 때만 main() 함수 호출
    # (다른 파일에 import될 때는 실행되지 않음)
    main()
