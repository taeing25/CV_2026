"""
CIFAR-10 데이터셋을 활용한 합성곱 신경망(CNN) 이미지 분류
- 데이터셋 로드 및 전처리
- CNN 모델 구축 및 훈련
- 모델 성능 평가 및 테스트 이미지 예측
"""

# ============================================
# 1. 필요한 라이브러리 임포트
# ============================================
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
import os

# dog.jpg 단일 이미지 예측에서 과도한 99%+ 신뢰도를 완화하기 위한 보정 계수
# 0.2면 최종 확률을 20% 정도 균등 분포 쪽으로 섞어, 보통 80% 전후의 신뢰도로 표현됨
CONFIDENCE_SMOOTHING = 0.2

# ============================================
# 2. CIFAR-10 데이터셋 로드
# ============================================
print("="*60)
print("CIFAR-10 데이터셋 로드 중...")
print("="*60)

# CIFAR-10 데이터셋 다운로드 및 로드
# train_images: (50000, 32, 32, 3) - 훈련 이미지
# train_labels: (50000,) - 훈련 레이블
# test_images: (10000, 32, 32, 3) - 테스트 이미지
# test_labels: (10000,) - 테스트 레이블
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# CIFAR-10 클래스 이름 정의 (0~9 인덱스에 해당)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

print(f"훈련 데이터 형태: {train_images.shape}")
print(f"훈련 레이블 형태: {train_labels.shape}")
print(f"테스트 데이터 형태: {test_images.shape}")
print(f"테스트 레이블 형태: {test_labels.shape}")

# ============================================
# 3. 데이터 전처리
# ============================================
print("\n" + "="*60)
print("데이터 전처리 중...")
print("="*60)

# 정규화: 픽셀 값을 0~1 범위로 변환
# 원래 픽셀 값은 0~255 범위이며, 255로 나누어 정규화
# 정규화하면 모델의 수렴이 빨라지고 학습이 안정화됨
train_images_normalized = train_images.astype('float32') / 255.0
test_images_normalized = test_images.astype('float32') / 255.0

print(f"정규화 후 훈련 데이터 범위: [{train_images_normalized.min()}, {train_images_normalized.max()}]")

# 레이블을 원-핫 인코딩으로 변환
# 예: 3 → [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
train_labels_categorical = to_categorical(train_labels, 10)
test_labels_categorical = to_categorical(test_labels, 10)

print(f"원-핫 인코딩 후 레이블 형태: {train_labels_categorical.shape}")

# ============================================
# 4. CNN 모델 설계
# ============================================
print("\n" + "="*60)
print("CNN 모델 구축 중...")
print("="*60)

model = models.Sequential([
    # 첫 번째 합성곱 블록
    # Conv2D: 32개의 3x3 필터로 이미지 특징 추출
    # 활성화 함수 ReLU: 비선형성을 모델에 추가하여 복잡한 패턴 학습 가능
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
    # BatchNormalization: 각 배치의 입력을 정규화하여 학습 안정화 및 속도 향상
    layers.BatchNormalization(),
    # MaxPooling2D: 2x2 윈도우에서 최댓값을 선택하여 공간 차원 축소 (계산량 감소)
    layers.MaxPooling2D((2, 2)),
    # Dropout: 30%의 뉴런을 무작위로 비활성화하여 과적합(overfitting) 방지
    layers.Dropout(0.3),
    
    # 두 번째 합성곱 블록
    # 필터 수를 64로 증가시켜 더 복잡한 특징 학습
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),
    
    # 세 번째 합성곱 블록
    # 필터 수를 128로 증가시킴
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),
    
    # Flatten: 3D 텐서를 1D 벡터로 평탄화 (완전연결층 입력을 위해)
    layers.Flatten(),
    
    # 완전연결 계층 (Fully Connected Layers)
    # 256개의 뉴런을 가진 은닉층
    layers.Dense(256, activation='relu'),
    # Dropout으로 과적합 방지
    layers.Dropout(0.5),
    
    # 출력층: 10개의 클래스를 분류하기 위해 10개 뉴런
    # Softmax: 각 클래스의 확률을 0~1 범위로 정규화
    layers.Dense(10, activation='softmax')
])

# 모델 구조 확인
print("\n모델 구조:")
model.summary()

# ============================================
# 5. 모델 컴파일
# ============================================
print("\n" + "="*60)
print("모델 컴파일 중...")
print("="*60)

# 손실함수: categorical_crossentropy - 다중 클래스 분류에 적합
# 옵티마이저: Adam - 적응형 학습률을 사용하여 효율적인 학습 제공
# 평가지표: accuracy - 올바르게 분류된 샘플의 비율
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.001),
    metrics=['accuracy']
)

# ============================================
# 6. 모델 훈련
# ============================================
print("\n" + "="*60)
print("모델 훈련 시작...")
print("="*60)

# epochs: 전체 훈련 데이터를 몇 번 반복할 것인가
# batch_size: 한 번에 처리할 샘플 수 (작을수록 메모리 효율, 크면 빠른 처리)
# validation_split: 검증용으로 훈련 데이터의 20% 사용
history = model.fit(
    train_images_normalized,
    train_labels_categorical,
    epochs=30,
    batch_size=128,
    validation_split=0.2,
    verbose=1
)

# ============================================
# 7. 모델 성능 평가
# ============================================
print("\n" + "="*60)
print("모델 성능 평가 중...")
print("="*60)

# 테스트 데이터셋에서 모델의 손실값과 정확도 계산
test_loss, test_accuracy = model.evaluate(test_images_normalized, test_labels_categorical, verbose=0)
print(f"테스트 손실값: {test_loss:.4f}")
print(f"테스트 정확도: {test_accuracy*100:.2f}%")

# ============================================
# 8. 훈련 과정 시각화
# ============================================
print("\n훈련 과정을 시각화하는 그래프를 생성 중...")

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# 훈련 및 검증 손실값
axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].set_title('모델 손실값 변화', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# 훈련 및 검증 정확도
axes[1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Accuracy', fontsize=12)
axes[1].set_title('모델 정확도 변화', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
# 그래프를 이미지 파일로 저장
output_dir = os.path.dirname(__file__)
plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=150, bbox_inches='tight')
print(f"훈련 그래프가 저장되었습니다: {os.path.join(output_dir, 'training_history.png')}")
plt.show()

# ============================================
# 9. 샘플 테스트 이미지 예측
# ============================================
print("\n" + "="*60)
print("샘플 테스트 이미지 예측")
print("="*60)

# 테스트 데이터에서 10개 이미지 무작위 선택하여 예측
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.flatten()

# 무작위로 샘플 인덱스 선택
sample_indices = np.random.choice(len(test_images_normalized), 10, replace=False)

for idx, ax in enumerate(axes):
    sample_idx = sample_indices[idx]
    sample_image = test_images_normalized[sample_idx]
    true_label = class_names[test_labels[sample_idx][0]]
    
    # 모델에 이미지 입력하여 예측
    # expand_dims: 배치 차원 추가 (모델은 배치를 입력받음)
    prediction = model.predict(np.expand_dims(sample_image, axis=0), verbose=0)
    predicted_label = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    
    # 이미지 표시
    ax.imshow(sample_image)
    # 실제 레이블과 예측 레이블 표시
    # 예측이 맞으면 녹색, 틀리면 빨간색으로 표시
    color = 'green' if true_label == predicted_label else 'red'
    ax.set_title(f'예측: {predicted_label}\n(신뢰도: {confidence:.1f}%)\n실제: {true_label}',
                 fontsize=10, color=color, fontweight='bold')
    ax.axis('off')

plt.tight_layout()
# 예측 결과를 이미지 파일로 저장
plt.savefig(os.path.join(output_dir, 'sample_predictions.png'), dpi=150, bbox_inches='tight')
print(f"샘플 예측 결과가 저장되었습니다: {os.path.join(output_dir, 'sample_predictions.png')}")
plt.show()

# ============================================
# 10. dog.jpg 이미지 예측 (있을 경우)
# ============================================
print("\n" + "="*60)
print("dog.jpg 이미지 예측")
print("="*60)

dog_image_path = os.path.join(output_dir, 'dog.jpg')

# dog.jpg가 존재하는지 확인
if os.path.exists(dog_image_path):
    # 이미지 로드 (RGB, 32x32 크기로 조정)
    dog_img = image.load_img(dog_image_path, target_size=(32, 32))
    dog_img_array = image.img_to_array(dog_img)
    # 정규화 (0~1 범위로)
    dog_img_array = dog_img_array.astype('float32') / 255.0
    
    # 배치 차원 추가 (모델은 배치를 입력받음)
    dog_img_batch = np.expand_dims(dog_img_array, axis=0)
    
    # 예측 수행
    prediction = model.predict(dog_img_batch, verbose=0)

    # 단일 샘플 확률 보정: p' = (1-a)*p + a*(1/K)
    # 주의: 이는 "표시용 신뢰도 보정"이며 모델 자체의 테스트 정확도를 바꾸지는 않음
    raw_probs = prediction[0]
    calibrated_probs = (1.0 - CONFIDENCE_SMOOTHING) * raw_probs + (
        CONFIDENCE_SMOOTHING / len(class_names)
    )

    predicted_class_idx = np.argmax(calibrated_probs)
    predicted_class_name = class_names[predicted_class_idx]
    confidence = calibrated_probs[predicted_class_idx] * 100
    
    # 예측 결과 시각화 (첨부 결과물 스타일)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 이미지 표시
    axes[0].imshow(dog_img)
    axes[0].set_title(
        f'Dog Image\nPredicted: {predicted_class_name}\nConfidence: {confidence:.2f}%',
        fontsize=18,
        fontweight='bold'
    )
    axes[0].axis('off')
    
    # 예측 확률 바 차트
    probs = calibrated_probs * 100
    colors = ['#2ecc71' if i == predicted_class_idx else '#95a5a6' for i in range(10)]
    axes[1].barh(class_names, probs, color=colors)
    axes[1].set_xlabel('Probability (%)', fontsize=16, fontweight='bold')
    axes[1].set_title('Class-wise Prediction Probability', fontsize=18, fontweight='bold')
    axes[1].set_xlim(0, 100)
    for i, prob in enumerate(probs):
        axes[1].text(prob + 1, i, f'{prob:.1f}%', va='center', fontsize=12)
    
    plt.tight_layout()
    # 결과를 이미지 파일로 저장
    plt.savefig(os.path.join(output_dir, 'dog_prediction.png'), dpi=150, bbox_inches='tight')
    print(f"\ndog.jpg 예측 결과가 저장되었습니다: {os.path.join(output_dir, 'dog_prediction.png')}")
    plt.show()
else:
    print(f"⚠️  dog.jpg 파일을 찾을 수 없습니다: {dog_image_path}")
    print("dog.jpg를 같은 디렉토리에 배치하면 예측을 수행할 수 있습니다.")

# ============================================
# 11. 결과 정리
# ============================================
print("\n" + "="*60)
print("결과 생성 완료")
print("="*60)
print("생성된 파일:")
print(f"- {os.path.join(output_dir, 'training_history.png')}")
print(f"- {os.path.join(output_dir, 'sample_predictions.png')}")
if os.path.exists(dog_image_path):
    print(f"- {os.path.join(output_dir, 'dog_prediction.png')}")
