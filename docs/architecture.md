# Architecture

> ⚠️ **Disclaimer** — 본 프로젝트는 개인 학습·연구 목적이며, 실제 의료 진단/임상 의사결정에 사용할 수 없습니다. 의료기기로 승인받지 않았습니다.

## 1. 개요

`medical_image_classification` 은 두 개의 의료 영상 분류 태스크를 단일 코드베이스로 학습·평가·추론하기 위한 TensorFlow/Keras 기반 파이프라인이다.

| Task | 데이터 | 클래스 | 입력 크기 | 라이선스 |
|---|---|---|---|---|
| `nih_cxr_binary` | NIH ChestX-ray14 | 2 (Pneumonia / Not) | 256×256 | Public Domain |
| `isic_multiclass` | ISIC 2019 | 7 (MEL/NV/BCC/AK/BKL/DF/VASC) | 224×224 | CC-BY-NC 4.0 |

설계 목표:

- **단일 GPU(RTX 4060 8GB) 친화** — mixed precision, gradient accumulation, OOM 자동 폴백
- **재현성** — 시드 고정, config 스키마 검증, run-별 디렉토리에 config 스냅샷 저장
- **2단계 학습** — head-only → fine-tune (BatchNorm 동결 옵션)
- **태스크별 일관 인터페이스** — manifest CSV(`filepath,label`) 기반 추상화

## 2. 레이어드 구조

```
┌──────────────────────────────────────────────────────────────────────┐
│  CLI Layer            main.py  (argparse, mode dispatch)             │
├──────────────────────────────────────────────────────────────────────┤
│  Orchestration        utils.bootstrap()  — Disclaimer / GPU /        │
│                       seed / mixed precision / XLA / logger          │
├──────────────────────────────────────────────────────────────────────┤
│  Use-case Layer       trainer.train()    evaluator.evaluate()        │
│                       predictor.predict()                            │
├──────────────────────────────────────────────────────────────────────┤
│  Domain Layer         model.build_model / unfreeze_for_finetune      │
│                       trainer.{focal_loss, GradAccumModel,           │
│                                 InnerWeightsCheckpoint}              │
├──────────────────────────────────────────────────────────────────────┤
│  Data Layer           data_processor.build_pipeline / build_datasets │
│                       dataset_downloader.{NIH, ISIC}                 │
├──────────────────────────────────────────────────────────────────────┤
│  Infrastructure       utils.{set_seed, get_logger, save_json,        │
│                              configure_mixed_precision, ...}         │
└──────────────────────────────────────────────────────────────────────┘
```

## 3. 모듈 책임

| 모듈 | 책임 | 주요 외부 인터페이스 |
|---|---|---|
| `main.py` | CLI 진입점, mode dispatch | `python main.py --mode {download,train,eval,predict}` |
| `utils.py` | Disclaimer, 시드, 로거, GPU 설정, **config 스키마 검증** | `bootstrap`, `load_config`, `validate_config`, `get_task_params` |
| `dataset_downloader.py` | 데이터셋 다운로드, manifest 생성 (NIH/ISIC), **lesion-level split** | `download_nih`, `download_isic`, `build_*_manifest` |
| `data_processor.py` | tf.data 파이프라인, CLAHE, 증강, class weight | `build_pipeline`, `build_datasets`, `compute_class_weights` |
| `model.py` | 백본 팩토리(EfficientNetB0/MobileNetV2/ResNet50), 헤드, 2단계 unfreeze | `build_model`, `unfreeze_for_finetune`, `get_preprocess_fn` |
| `trainer.py` | 2-stage 학습 루프, focal loss, gradient accumulation, 콜백 | `train`, `binary/categorical_focal_loss`, `GradAccumModel` |
| `evaluator.py` | test set 메트릭 계산, confusion matrix, ROC | `evaluate` |
| `predictor.py` | 단일/배치 추론, **Grad-CAM 시각화**, 워터마크 | `predict`, `predict_one`, `grad_cam` |

## 4. 런타임 흐름

### 4.1 학습 (`--mode train`)

```
main.py
  └── utils.bootstrap()                      # Disclaimer/시드/GPU/MP/XLA
        └── utils.load_config()              # ← validate_config()
  └── trainer.train(config, logger)
        ├── data_processor.build_datasets()  # train/val/test + class weights
        ├── model.build_model()              # base frozen
        ├── [optional] GradAccumModel(...)   # accum_steps>1일 때만 wrap
        │
        ├── Stage1: head-only  fit(epochs_head, lr_head)
        │     callbacks: ckpt(val_auc), early_stop, ReduceLR, TB, CSV, GradAccumFlush
        │
        ├── model.unfreeze_for_finetune()    # fine_tune_at, freeze_bn
        ├── re-compile with lr_finetune
        │
        ├── Stage2: fine-tune fit(epochs_finetune)
        │
        └── 체크포인트 fallback: stage2/best → stage1/best → final
```

### 4.2 추론 (`--mode predict`)

```
main.py → predictor.predict()
  ├── load_model_for_inference()  # build_model + load_weights
  ├── _load_and_prepare()         # cv2 read → CLAHE? → resize → backbone preprocess
  ├── model.predict_on_batch()
  └── grad_cam() (Conv2D auto-detect) → overlay → watermark → PNG
```

### 4.3 데이터 파이프라인

```
manifest CSV  ──▶ tf.data.Dataset.from_tensor_slices
                        │
                        ├── shuffle (train only)
                        ├── decode_image (uint8)
                        ├── (X-ray) CLAHE via py_function
                        ├── resize → float32
                        ├── augment (X-ray: brightness/contrast/zoom/rotate / Skin: flip+jitter)
                        ├── backbone preprocess_input
                        └── one-hot or sigmoid label → batch → prefetch
```

## 5. 데이터 스플릿 전략

- **NIH**: 공식 `train_val_list.txt` / `test_list.txt`. train_val 내부에서 `train_test_split` 으로 10% val 분리(stratified).
- **ISIC**: `ISIC_2019_Training_Metadata.csv` 의 `lesion_id` 가 존재하면 `StratifiedGroupKFold` 로 lesion-level 분리(누수 방지). 메타데이터 부재 시 image-level fallback 후 경고.

## 6. VRAM 최적화 (RTX 4060 8GB 기준)

| 기법 | 설정 키 | 효과 |
|---|---|---|
| Mixed precision (fp16) | `mixed_precision: true` | VRAM ~40% 절감 |
| XLA JIT | `xla_jit: true` | 커널 fusion |
| Gradient accumulation | `grad_accum_steps: N` | 유효 배치 = `batch_size × N` |
| Memory growth | `memory_growth: true` | 전체 VRAM 선점 방지 |
| OOM 자동 폴백 | `_build_with_oom_retry` | batch_size 절반 재시도 (최대 2회) |

## 7. Config 스키마 검증

`utils.load_config()` 는 호출 시 자동으로 `validate_config()` 를 실행한다. 검증 항목:

- 필수 키 존재 (`batch_size`, `lr_head`, `task`, …)
- 타입 (int/float/bool/str), bool ↔ int 혼동 거부
- 수치 범위 (예: `dropout ∈ [0, 0.95]`, `lr ∈ (0, 1]`)
- 카테고리 값 (`task`, `backbone`, `loss`, `optimizer`)
- `paths` 하위 필수 키 (`nih_dir`, `isic_dir`, `models_dir`, …)
- `isic_classes` 길이 = `num_classes_isic`

검증 실패 시 `ValueError` 와 함께 모든 위반 항목을 한 번에 보고한다.

## 8. 디렉토리 레이아웃

```
medical_image_classification/
├── main.py                        # CLI 진입점
├── config.json                    # 단일 설정 파일 (validate_config로 검증)
├── utils.py                       # bootstrap / 시드 / config 검증
├── dataset_downloader.py          # NIH / ISIC 다운로드 & manifest
├── data_processor.py              # tf.data 파이프라인
├── model.py                       # 백본 팩토리
├── trainer.py                     # 2-stage 학습 + GradAccum
├── evaluator.py                   # 평가 메트릭
├── predictor.py                   # 추론 + Grad-CAM
├── tests/                         # pytest 단위 테스트 (39개)
│   ├── test_config_validation.py
│   ├── test_data_processor.py
│   ├── test_losses.py
│   ├── test_model.py
│   └── test_utils.py
├── docs/                          # 본 문서
│   ├── architecture.md
│   └── uml.md
├── dataset/                       # 다운로드 데이터셋 (gitignore)
├── models/                        # 학습 산출물 (run_dir/)
├── logs/                          # 학습/다운로드 로그
└── results/                       # 평가/추론 산출물
```

`run_dir = models/<task>_<backbone>_<timestamp>/` 하위 구조:

```
run_dir/
├── run_config.json        # config + meta + task_params 스냅샷
├── stage1/
│   ├── best.weights.h5
│   ├── history.csv
│   └── tb/                # TensorBoard
├── stage2/
│   ├── best.weights.h5
│   ├── history.csv
│   └── tb/
├── final.weights.h5       # 최종 가중치 (fallback용)
└── history.json           # stage1+stage2 통합 학습 곡선
```

## 9. 확장 포인트

| 추가하고 싶은 것 | 손대야 할 곳 |
|---|---|
| 새로운 백본 | `model._BACKBONES` 에 등록 + `validate_config._VALID_BACKBONES` 갱신 |
| 새로운 태스크 | `utils.get_task_params` + `dataset_downloader.build_*_manifest` + `_VALID_TASKS` |
| 새로운 augmentation | `data_processor._augment_*` |
| 새로운 loss | `trainer.build_loss` 분기 + `_VALID_LOSSES` 추가 |
| 새로운 메트릭 | `trainer.build_metrics`, `evaluator` 의 계산 블록 |
| 분산 학습 지원 | `trainer.GradAccumModel` 의 단일 GPU 가드 해제 + `tf.distribute` 호환 누적 구현 필요 |

## 10. 알려진 제약

- `GradAccumModel` 은 **단일 GPU 전용** (생성자에서 `num_replicas_in_sync > 1` 이면 `RuntimeError`).
- NIH Pneumonia 라벨은 텍스트 마이닝 기반 weak label → AUROC 0.6 내외가 현실적 상한.
- ISIC GroundTruth 는 7-class만 사용 (SCC 제외, UNK 행 제거).
- DICOM/NIfTI 미지원 (예측은 `.png/.jpg/.jpeg`만).
- TensorFlow Addons 미설치 시 X-ray 회전 증강 자동 비활성 (런타임 1회 경고).
