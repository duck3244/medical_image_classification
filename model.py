"""
model.py
백본 팩토리 - EfficientNetB0 / MobileNetV2 / ResNet50 지원.
- ImageNet 사전학습 가중치 (Apache 2.0 / MIT)
- 2단계 학습(head-only → fine-tune) 지원
- Mixed precision 환경에서 최종 Dense는 float32 유지 (수치 안정성)
- 파인튜닝 시 BatchNorm 동결 옵션
"""

from typing import Optional, Tuple

import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2, ResNet50


# ======================================================================
# 백본 레지스트리
# ======================================================================

_BACKBONES = {
    "EfficientNetB0": {
        "cls": EfficientNetB0,
        "preprocess": tf.keras.applications.efficientnet.preprocess_input,
    },
    "MobileNetV2": {
        "cls": MobileNetV2,
        "preprocess": tf.keras.applications.mobilenet_v2.preprocess_input,
    },
    "ResNet50": {
        "cls": ResNet50,
        "preprocess": tf.keras.applications.resnet50.preprocess_input,
    },
}


def get_preprocess_fn(backbone: str):
    """백본별 ImageNet 전처리 함수를 반환 (data_processor에서 사용)."""
    if backbone not in _BACKBONES:
        raise ValueError(f"Unknown backbone: {backbone}")
    return _BACKBONES[backbone]["preprocess"]


# ======================================================================
# 모델 빌더
# ======================================================================

def build_model(
    backbone: str = "EfficientNetB0",
    num_classes: int = 2,
    img_size: int = 256,
    dropout: float = 0.5,
    l2_reg: float = 1e-4,
    weights: Optional[str] = "imagenet",
) -> Tuple[Model, Model]:
    """
    분류 모델 생성.

    Returns:
        (model, base_model) - base_model은 단계적 unfreeze 제어용으로 반환.

    주의:
        - base_model은 기본적으로 freeze된 상태로 반환 (1단계 head-only 학습용).
        - Mixed precision 환경에서도 최종 softmax/sigmoid는 float32로 강제하여
          loss 수치 안정성을 확보.
    """
    if backbone not in _BACKBONES:
        raise ValueError(
            f"Unknown backbone: {backbone}. Choose from {list(_BACKBONES.keys())}"
        )

    input_shape = (img_size, img_size, 3)
    inputs = layers.Input(shape=input_shape, name="image")

    base_cls = _BACKBONES[backbone]["cls"]
    base_model = base_cls(
        include_top=False,
        weights=weights,
        input_tensor=inputs,
        pooling=None,
    )
    base_model.trainable = False  # 1단계: head-only 학습

    x = base_model.output
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Dropout(dropout, name="head_dropout")(x)

    kreg = regularizers.l2(l2_reg) if l2_reg and l2_reg > 0 else None
    is_binary = num_classes == 2
    if is_binary:
        x = layers.Dense(1, name="logits", kernel_regularizer=kreg)(x)
        outputs = layers.Activation("sigmoid", dtype="float32", name="pred")(x)
    else:
        x = layers.Dense(num_classes, name="logits", kernel_regularizer=kreg)(x)
        outputs = layers.Activation("softmax", dtype="float32", name="pred")(x)

    model = Model(inputs=inputs, outputs=outputs, name=f"{backbone}_head{num_classes}")
    return model, base_model


# ======================================================================
# 2단계 파인튜닝 제어
# ======================================================================

def unfreeze_for_finetune(
    base_model: Model,
    fine_tune_at: int = 150,
    freeze_bn: bool = True,
):
    """
    파인튜닝을 위해 base_model의 일부 레이어를 unfreeze.

    Args:
        base_model: build_model이 반환한 base_model
        fine_tune_at: 이 인덱스 이전의 레이어는 freeze 유지
        freeze_bn: True면 BatchNorm 레이어는 unfreeze 대상에서 제외
                   (소규모 배치에서 running stats 흔들림 방지)
    """
    base_model.trainable = True
    total = len(base_model.layers)
    fine_tune_at = max(0, min(fine_tune_at, total - 1))

    for i, layer in enumerate(base_model.layers):
        if i < fine_tune_at:
            layer.trainable = False
        else:
            if freeze_bn and isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = True

    trainable = sum(1 for l in base_model.layers if l.trainable)
    return {
        "total_layers": total,
        "trainable_layers": trainable,
        "fine_tune_at": fine_tune_at,
        "freeze_bn": freeze_bn,
    }


def count_params(model: Model) -> dict:
    """학습 가능/전체 파라미터 수."""
    trainable = sum(
        tf.keras.backend.count_params(w) for w in model.trainable_weights
    )
    non_trainable = sum(
        tf.keras.backend.count_params(w) for w in model.non_trainable_weights
    )
    return {
        "trainable": int(trainable),
        "non_trainable": int(non_trainable),
        "total": int(trainable + non_trainable),
    }


# ======================================================================
# Smoke test
# ======================================================================

if __name__ == "__main__":
    for bb in ["EfficientNetB0", "MobileNetV2"]:
        print(f"\n=== {bb} ===")
        model, base = build_model(bb, num_classes=2, img_size=256)
        print("Stage1 (head-only):", count_params(model))
        info = unfreeze_for_finetune(base, fine_tune_at=150, freeze_bn=True)
        print("Stage2 (fine-tune):", count_params(model), info)
        dummy = tf.zeros((1, 256, 256, 3))
        out = model(dummy, training=False)
        print("Output shape:", out.shape)
