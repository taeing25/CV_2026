"""MNIST 손글씨 숫자 분류기 예제.

이 스크립트는 TensorFlow/Keras를 사용해서 다음 순서로 동작합니다.
1. MNIST 데이터셋을 로드합니다.
2. 학습 데이터와 테스트 데이터를 준비합니다.
3. 간단한 완전연결 신경망(Sequential + Dense) 모델을 구성합니다.
4. 모델을 학습시키고 테스트 정확도를 평가합니다.

MNIST 이미지는 28x28 크기의 흑백 이미지이므로,
입력값을 0~255 범위에서 0~1 범위로 정규화해서 모델 학습을 안정화합니다.
"""

from __future__ import annotations

import numpy as np
# from tensorflow.keras.datasets import mnist
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Flatten, Dense
import tensorflow as tf

def load_and_prepare_data():
    """MNIST 데이터를 불러오고 모델 입력에 맞게 전처리한다.

    Returns:
        x_train: 학습 이미지 데이터
        y_train: 학습 정답 레이블
        x_test: 테스트 이미지 데이터
        y_test: 테스트 정답 레이블
    """
    # MNIST는 기본적으로 train/test로 나뉘어 제공된다.
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # 픽셀 값은 0~255 정수이므로, 0~1 실수로 바꿔 학습이 잘 되도록 만든다.
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    return x_train, y_train, x_test, y_test


def build_model():
    """손글씨 숫자를 분류하는 간단한 신경망을 만든다.

    모델 구조:
    - Flatten: 28x28 이미지를 784차원 벡터로 펼침
    - Dense(128, relu): 은닉층에서 특징을 학습
    - Dense(10, softmax): 0~9 숫자 중 하나의 확률을 출력
    """
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )

    # 다중 분류 문제이므로 sparse_categorical_crossentropy를 사용한다.
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def main():
    # 1) 데이터 로드 및 전처리
    x_train, y_train, x_test, y_test = load_and_prepare_data()

    # 2) 모델 생성
    model = build_model()

    # 3) 모델 학습
    # epochs는 학습 반복 횟수다. 너무 크지 않게 잡아 빠르게 실행되도록 한다.
    history = model.fit(
        x_train,
        y_train,
        epochs=5,
        batch_size=32,
        validation_split=0.1,
        verbose=1,
    )

    # 4) 테스트 데이터로 최종 성능 평가
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

    print("\n학습 완료")
    print(f"테스트 손실: {test_loss:.4f}")
    print(f"테스트 정확도: {test_accuracy:.4f}")

    # 5) 예측 예시 확인
    # 테스트 이미지 1개를 가져와 모델이 어떤 숫자로 예측하는지 확인한다.
    sample_image = x_test[0:1]
    prediction = model.predict(sample_image, verbose=0)
    predicted_label = int(np.argmax(prediction[0]))

    print(f"실제 레이블: {y_test[0]}")
    print(f"예측 레이블: {predicted_label}")

    # 학습 과정도 간단히 출력하면, 손실과 정확도가 어떻게 바뀌었는지 확인할 수 있다.
    print("\n학습 기록 예시:")
    print(f"- 마지막 epoch 학습 정확도: {history.history['accuracy'][-1]:.4f}")
    print(f"- 마지막 epoch 검증 정확도: {history.history['val_accuracy'][-1]:.4f}")


if __name__ == "__main__":
    main()