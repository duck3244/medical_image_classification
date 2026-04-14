# Medical Image Classification (Learning Project)

> ## ⚠️ DISCLAIMER — 반드시 읽어주세요
>
> **본 프로젝트는 개인 학습·연구 목적으로만 제작되었습니다.**
> - 실제 의료 진단, 임상 의사결정, 환자 관리에 **사용할 수 없습니다**.
> - 의료기기(MFDS/FDA 등)로 승인받지 않았습니다.
> - 본 프로젝트의 예측 결과는 **의학적 소견이 아니며**, 어떠한 의료 행위의 근거로도 사용될 수 없습니다.
> - 실제 증상이 있는 경우 반드시 의료 전문가의 진료를 받으십시오.
>
> **This project is for personal educational and research purposes only.**
> It is NOT a medical device and MUST NOT be used for diagnosis or clinical decisions.

---

## 개요

공개 의료 영상 데이터셋을 활용한 경량 분류 모델 학습 프로젝트. 기존 비전 프로젝트
경험을 의료 도메인으로 확장하며, RTX 4060 8GB 환경에서 학습 가능하도록 메모리
최적화를 적용한다.

## 지원 태스크

| Task | 데이터셋 | 분류 | 라이선스 |
|---|---|---|---|
| `nih_cxr_binary` | NIH ChestX-ray14 | Pneumonia 이진 | Public Domain |
| `isic_multiclass` | ISIC 2019 | 피부병변 7-class | CC-BY-NC 4.0 (비상업) |

## 백본 모델 (모두 Apache 2.0 / MIT)

- **EfficientNetB0** (기본 권장, ~5.3M params)
- MobileNetV2 (~3.5M params)
- ResNet50 (~25M params)

## 환경

- Python 3.9+
- TensorFlow 2.10 (GPU)
- CUDA 11.2 / cuDNN 8.1
- GPU: RTX 4060 8GB 기준 최적화

## 설치

```bash
pip install -r requirements.txt
```

## 사용법

```bash
# 1. 데이터셋 다운로드 (라이선스 동의 필요)
python main.py --mode download --task nih_cxr_binary --agree
python main.py --mode download --task isic_multiclass --agree

# 2. 학습
python main.py --mode train --task nih_cxr_binary --backbone EfficientNetB0

# 3. 평가
python main.py --mode eval --task nih_cxr_binary --weights models/best.h5

# 4. 추론 (Grad-CAM 포함)
python main.py --mode predict --image sample.png --weights models/best.h5
```

## RTX 4060 8GB 기본 프로파일

| 항목 | 값 |
|---|---|
| 이미지 크기 | 256 (NIH) / 224 (ISIC) |
| Batch size | 16 × accum 2 = 유효 32 |
| Mixed precision / XLA | fp16 / on |
| 예상 VRAM | ~5–6 GB |

OOM 시 `batch_size=8` 또는 `gradient_checkpointing=true`.

## 학습 안정화 옵션 (config.json)

| 키 | 기본 | 설명 |
|---|---|---|
| `oversample` | true | binary task에서 양성/음성 50:50 sample_from_datasets |
| `focal_alpha_auto` | true | 클래스 분포 기반 focal α 자동 산출 |
| `focal_gamma` | 3.0 | 불균형이 심할수록 ↑ (2.0–4.0) |
| `dropout` | 0.5 | head dropout |
| `l2_reg` | 1e-4 | Dense logits L2 정규화 |
| `cache_dataset` | false | 메모리 여유 시 tf.data cache |
| `fine_tune_at` | 200 | 값이 작을수록 공격적 unfreeze |

## 데이터셋 라이선스

- **NIH ChestX-ray14**: U.S. NIH Clinical Center 공개, Public Domain.
  https://nihcc.app.box.com/v/ChestXray-NIHCC
- **ISIC 2019**: CC-BY-NC 4.0, 비상업적 연구 목적만 허용.
  https://challenge.isic-archive.com/data/#2019

다운로드 스크립트 실행 시 라이선스 고지문 출력 후 `--agree` 플래그로 동의를 확인합니다.

## 프로젝트 구조

`main.py`(CLI) → `dataset_downloader` / `data_processor` / `model` / `trainer` / `evaluator` / `predictor`. 산출물은 `models/`, `logs/`, `results/`.

## 알려진 한계

- NIH Pneumonia 이진은 라벨 노이즈(텍스트 마이닝 weak label)와 양성률 ~1% 불균형으로 AUROC 0.6 내외가 현실적 상한. 실험 용도로만 사용.
- ISIC split은 `ISIC_2019_Training_Metadata.csv` 의 `lesion_id`가 있을 때 lesion-level grouped split을 적용해 동일 lesion의 이미지가 split 사이에 흩어지지 않도록 보장(`dataset_downloader._isic_lesion_group_split`). 메타데이터가 없으면 image-level fallback이며 이 경우 메트릭이 낙관적으로 편향될 수 있음.

## 라이선스

본 프로젝트 소스 코드는 학습 목적 공개. 데이터셋과 사전학습 가중치는 각자의 원본
라이선스를 따릅니다 (`NOTICE` 참조).
