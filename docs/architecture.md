# Architecture

> ⚠️ **Disclaimer** — 본 프로젝트는 개인 학습·연구 목적이며, 실제 의료 진단/임상 의사결정에 사용할 수 없습니다. 의료기기로 승인받지 않았습니다. NOT a medical device. NOT for diagnosis.

## 1. 개요

`medical_image_classification` 은 두 개의 의료 영상 분류 태스크(NIH ChestX-ray14 / ISIC 2019)를 **TensorFlow/Keras 학습 파이프라인 + FastAPI 추론 서비스 + React 단일사용자 MVP UI** 로 구성한 풀스택 학습 프로젝트다.

| Task | 데이터 | 클래스 | 입력 크기 | 라이선스 |
|---|---|---|---|---|
| `nih_cxr_binary` | NIH ChestX-ray14 | 2 (Pneumonia / Not) | 256×256 | Public Domain |
| `isic_multiclass` | ISIC 2019 | 7 (MEL/NV/BCC/AK/BKL/DF/VASC) | 224×224 | CC-BY-NC 4.0 |

설계 목표:

- **단일 GPU(RTX 4060 8GB) 친화** — mixed precision, gradient accumulation, OOM 자동 폴백
- **재현성** — 시드 고정, config 스키마 검증, run-별 디렉토리에 config 스냅샷 저장
- **2단계 학습** — head-only → fine-tune (BatchNorm 동결 옵션)
- **태스크별 일관 인터페이스** — manifest CSV(`filepath,label`) 기반 추상화
- **즉시 사용 가능한 데모** — 학습 가중치 부재 시 합성 mini 데이터로 자동 부트스트랩
- **단일 사용자 MVP UI** — FastAPI + React + Vite, Grad-CAM 시각화 포함

## 2. 시스템 구성 (Top-level)

```
┌──────────────────────────────────────────────────────────────────────────┐
│  Browser (single user)                                                    │
│   ├─ React 18 + Vite 5 + Tailwind 3                                       │
│   ├─ TanStack Query (health polling / models / predict mutation)         │
│   └─ React Router v6  (/predict, /models)                                │
└──────────────────────────────────────────────────────────────────────────┘
                              │ HTTP /api/*  /static/results/*
                              ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  FastAPI (backend/app.py)                                                 │
│   ├─ lifespan: load_config → GPU memory_growth → find/auto-bootstrap     │
│   │            → warmup (단일 GPU 락 + 모델 캐시)                        │
│   ├─ Routers:  /api/health  /api/tasks  /api/models  /api/predict        │
│   └─ StaticFiles: /static/results → backend/results/                     │
└──────────────────────────────────────────────────────────────────────────┘
                              │ services/* (인프로세스 호출)
                              ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  ML Pipeline (TensorFlow 2.10)                                            │
│   ├─ CLI:        backend/main.py (download/train/eval/predict)           │
│   ├─ Use-case:   trainer.py / evaluator.py / predictor.py                │
│   ├─ Domain:     model.py (backbones), focal_loss, GradAccumModel        │
│   ├─ Data:       data_processor.py (tf.data), dataset_downloader.py      │
│   └─ Infra:      utils.py (seed/GPU/config validate/logger)              │
└──────────────────────────────────────────────────────────────────────────┘
                              │ filesystem
                              ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  Artifacts                                                                │
│   dataset/   models/<run_dir>/  logs/  results/<predict_run>/  uploads/  │
└──────────────────────────────────────────────────────────────────────────┘
```

## 3. 레이어드 구조 (백엔드)

```
┌──────────────────────────────────────────────────────────────────────┐
│  Presentation     React SPA (frontend/src)                            │
│                   pages: Predict, Models   layout: DisclaimerBanner  │
├──────────────────────────────────────────────────────────────────────┤
│  HTTP API         FastAPI routers (backend/api/routes/*.py)          │
│                   health / tasks / models / predict                  │
├──────────────────────────────────────────────────────────────────────┤
│  Service          backend/services/                                  │
│                   inference (model cache + asyncio.Lock)             │
│                   bootstrap (auto synthetic training)                │
├──────────────────────────────────────────────────────────────────────┤
│  CLI Entry        backend/main.py (argparse, mode dispatch)          │
├──────────────────────────────────────────────────────────────────────┤
│  Orchestration    utils.bootstrap() — Disclaimer / GPU /             │
│                   seed / mixed precision / XLA / logger              │
├──────────────────────────────────────────────────────────────────────┤
│  Use-case Layer   trainer.train()   evaluator.evaluate()             │
│                   predictor.predict() / predict_one()                │
├──────────────────────────────────────────────────────────────────────┤
│  Domain Layer     model.build_model / unfreeze_for_finetune          │
│                   trainer.{focal_loss, GradAccumModel,               │
│                            InnerWeightsCheckpoint}                   │
├──────────────────────────────────────────────────────────────────────┤
│  Data Layer       data_processor.build_pipeline / build_datasets     │
│                   dataset_downloader.{NIH, ISIC}                     │
├──────────────────────────────────────────────────────────────────────┤
│  Infrastructure   utils.{set_seed, get_logger, save_json,            │
│                          configure_mixed_precision, validate_config} │
└──────────────────────────────────────────────────────────────────────┘
```

## 4. 모듈 책임

### 4.1 백엔드 (Python)

| 모듈 | 책임 | 주요 외부 인터페이스 |
|---|---|---|
| `app.py` | FastAPI 진입점, lifespan (config/GPU/auto-bootstrap/warmup), CORS, StaticFiles | `uvicorn app:app` |
| `api/routes/health.py` | GET /api/health — TF/GPU/모델 상태 + disclaimer | `health()` |
| `api/routes/tasks.py` | GET /api/tasks — 지원 태스크 메타 + 사용가능 백본 | `list_tasks()` |
| `api/routes/models.py` | GET /api/models — `backend/models/**/*.weights.h5` 스캔 | `list_models()` |
| `api/routes/predict.py` | POST /api/predict — multipart 업로드, 크기/확장자 검증, Grad-CAM URL 반환 | `predict_image()` |
| `services/inference.py` | 모델 캐시 `(task, backbone, weights)` + `asyncio.Lock` 직렬화 + fp32 정책 강제 | `ensure_model`, `warmup`, `run_predict_one`, `find_default_weights` |
| `services/bootstrap.py` | 가중치 부재 시 합성 mini CXR로 1+1 epoch 자동 학습 (데모용) | `auto_bootstrap`, `run_bootstrap_training` |
| `main.py` | CLI 진입점 (download/train/eval/predict) | `python main.py --mode ...` |
| `utils.py` | Disclaimer, 시드, 로거, GPU 설정, **config 스키마 검증** | `bootstrap`, `load_config`, `validate_config`, `get_task_params`, `setup_gpu_memory_growth` |
| `dataset_downloader.py` | 데이터셋 다운로드, manifest 생성 (NIH/ISIC), **lesion-level split** | `download_nih`, `download_isic`, `build_*_manifest` |
| `data_processor.py` | tf.data 파이프라인, CLAHE, 증강, class weight | `build_pipeline`, `build_datasets`, `compute_class_weights` |
| `model.py` | 백본 팩토리(EfficientNetB0/MobileNetV2/ResNet50), 헤드, 2단계 unfreeze | `build_model`, `unfreeze_for_finetune`, `get_preprocess_fn` |
| `trainer.py` | 2-stage 학습 루프, focal loss, gradient accumulation, 콜백 | `train`, `binary/categorical_focal_loss`, `GradAccumModel` |
| `evaluator.py` | test set 메트릭 계산, confusion matrix, ROC | `evaluate` |
| `predictor.py` | 단일/배치 추론, **Grad-CAM 시각화**, 워터마크 | `predict`, `predict_one`, `grad_cam` |

### 4.2 프론트엔드 (TypeScript / React)

| 파일 | 책임 |
|---|---|
| `src/main.tsx` | 엔트리. QueryClientProvider + BrowserRouter |
| `src/App.tsx` | Routes — `/predict`, `/models`, fallback → `/predict` |
| `src/components/Layout.tsx` | 헤더/푸터, 네비, **health 폴링 dot**(15s), Outlet |
| `src/components/DisclaimerBanner.tsx` | 모든 페이지 상단 영구 경고 배너 |
| `src/components/DisclaimerModal.tsx` | 최초 방문 시 동의 모달 |
| `src/pages/PredictPage.tsx` | 파일 업로드 + task/backbone/weights override + Grad-CAM 결과 표시 |
| `src/pages/ModelsPage.tsx` | `backend/models` 스캔 결과 테이블 |
| `src/api/client.ts` | fetch 래퍼 (`/api`, `/static` same-origin via Vite proxy) |
| `src/api/types.ts` | 백엔드 응답 스키마 (HealthResponse / TasksResponse / ModelsResponse / PredictResponse) |

## 5. HTTP API 표면

| Method | Path | 요청 | 응답 요약 |
|---|---|---|---|
| GET | `/api/health` | — | `{status, disclaimer, tensorflow, gpu_*, current_task/backbone, loaded_model_key, default_weights}` |
| GET | `/api/tasks` | — | 지원 task 목록(클래스/img_size/라이선스) + `available_backbones` |
| GET | `/api/models` | — | `runs[].weights[]` (stage1/stage2/final, size_mb, path) + `default_weights` |
| POST | `/api/predict` | multipart: `file`, optional `task`, `backbone`, `weights` | 예측 클래스/확률/Grad-CAM URL + disclaimer |
| GET | `/static/results/<run>/<file>` | — | Grad-CAM PNG (results/ 정적 서빙). uploads/ 는 **노출 안 함** |

업로드 제약: PNG/JPG/JPEG, 10 MB 이하. CORS는 dev 한정(`localhost:5173`).

## 6. 런타임 흐름

### 6.1 서버 부팅 (`uvicorn app:app`)

```
lifespan(startup)
  ├── _make_logger()                       # 콘솔 로거
  ├── load_config(backend/config.json)     # ← validate_config()
  ├── setup_gpu_memory_growth()
  ├── find_default_weights(models/)
  │   └── 없음 → auto_bootstrap()
  │              ├── generate_synthetic_dataset (60장 합성 CXR)
  │              └── trainer.train(boot_cfg)  # 1+1 epoch, batch=8, MP off
  ├── app.state.default_weights = <path>
  └── warmup() — services.inference.ensure_model() 사전 로드
```

### 6.2 예측 (POST /api/predict)

```
predict_image()
  ├── _validate_upload(file)                          # ext / size
  ├── _build_request_config(app_cfg, task, backbone)  # override 머지
  ├── get_task_params(cfg)                            # num_classes/img_size 검증
  ├── _save_upload → backend/uploads/<uuid>.<ext>
  ├── run_predict_one()
  │     └── ensure_model(cfg, weights)                # 캐시 키 다르면 reload
  │           └── asyncio.Lock + asyncio.to_thread    # Keras non-thread-safe 가드
  │                 └── predictor.predict_one()
  │                       ├── _load_and_prepare       # cv2 → CLAHE? → resize → preprocess
  │                       ├── model.predict_on_batch
  │                       └── grad_cam → overlay → watermark → PNG
  └── relative gradcam_path → /static/results/...    # 응답 URL
```

### 6.3 학습 (`python main.py --mode train`)

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

### 6.4 데이터 파이프라인 (한 샘플)

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

## 7. 동시성 / 스레드 안전성

- FastAPI는 sync handler를 스레드풀에서 실행. Keras 모델은 thread-safe가 아니라 `services.inference._lock` (`asyncio.Lock`) 으로 추론을 **직렬화**.
- 모델 캐시 키는 `(task, backbone, weights_path)`. 키가 바뀌면 lock 안에서 lazy reload.
- TF 연산은 `asyncio.to_thread()` 로 디스패치하여 이벤트 루프 블로킹 방지.
- mixed_precision 정책이 `mixed_*` 면 서비스 진입 시 강제 `float32` (Grad-CAM 안정성).

## 8. 데이터 스플릿 전략

- **NIH**: 공식 `train_val_list.txt` / `test_list.txt`. train_val 내부에서 `train_test_split` 으로 10% val 분리(stratified).
- **ISIC**: `ISIC_2019_Training_Metadata.csv` 의 `lesion_id` 가 존재하면 `StratifiedGroupKFold` 로 lesion-level 분리(누수 방지). 메타데이터 부재 시 image-level fallback 후 경고.
- **Bootstrap (합성)**: 40/10/10 = 60장. 양성에 우폐 patch — UI 동작 검증 전용, 임상적 의미 없음.

## 9. VRAM 최적화 (RTX 4060 8GB 기준)

| 기법 | 설정 키 | 효과 |
|---|---|---|
| Mixed precision (fp16) | `mixed_precision: true` | VRAM ~40% 절감 |
| XLA JIT | `xla_jit: true` | 커널 fusion |
| Gradient accumulation | `grad_accum_steps: N` | 유효 배치 = `batch_size × N` |
| Memory growth | `memory_growth: true` | 전체 VRAM 선점 방지 |
| OOM 자동 폴백 | `_build_with_oom_retry` | batch_size 절반 재시도 (최대 2회) |

## 10. Config 스키마 검증

`utils.load_config()` 는 호출 시 자동으로 `validate_config()` 를 실행한다. 검증 항목:

- 필수 키 존재 (`batch_size`, `lr_head`, `task`, …)
- 타입 (int/float/bool/str), bool ↔ int 혼동 거부
- 수치 범위 (예: `dropout ∈ [0, 0.95]`, `lr ∈ (0, 1]`)
- 카테고리 값 (`task`, `backbone`, `loss`, `optimizer`)
- `paths` 하위 필수 키 (`nih_dir`, `isic_dir`, `models_dir`, …)
- `isic_classes` 길이 = `num_classes_isic`

검증 실패 시 `ValueError` 와 함께 모든 위반 항목을 한 번에 보고한다.

## 11. 디렉토리 레이아웃

```
medical_image_classification/
├── README.md / LICENSE / NOTICE
├── docs/                              # 본 문서 (루트)
│   ├── architecture.md
│   └── uml.md
├── backend/
│   ├── app.py                         # FastAPI 진입점
│   ├── main.py                        # CLI 진입점
│   ├── config.json                    # 단일 설정 파일 (validate_config로 검증)
│   ├── api/
│   │   └── routes/
│   │       ├── health.py
│   │       ├── tasks.py
│   │       ├── models.py
│   │       └── predict.py
│   ├── services/
│   │   ├── inference.py               # 모델 캐시 + lock + warmup
│   │   └── bootstrap.py               # 합성 데이터 자동 학습
│   ├── utils.py                       # bootstrap / 시드 / config 검증
│   ├── dataset_downloader.py
│   ├── data_processor.py
│   ├── model.py
│   ├── trainer.py
│   ├── evaluator.py
│   ├── predictor.py
│   ├── requirements.txt
│   ├── tests/                         # pytest 단위 테스트
│   ├── docs/                          # 백엔드 전용 상세 문서 (참고용)
│   ├── dataset/                       # 다운로드 데이터셋 (gitignore)
│   ├── models/                        # 학습 산출물 (run_dir/)
│   ├── uploads/                       # 업로드 임시 저장 (정적 노출 X)
│   ├── logs/                          # 학습/다운로드 로그
│   └── results/                       # 평가/추론 산출물 → /static/results
└── frontend/
    ├── package.json
    ├── vite.config.ts                 # /api, /static proxy → :8000
    ├── index.html
    └── src/
        ├── main.tsx / App.tsx
        ├── components/                # Layout, DisclaimerBanner/Modal
        ├── pages/                     # PredictPage, ModelsPage
        └── api/                       # client.ts, types.ts
```

`run_dir = backend/models/<task>_<backbone>_<timestamp>/` 하위 구조:

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

## 12. 확장 포인트

| 추가하고 싶은 것 | 손대야 할 곳 |
|---|---|
| 새로운 백본 | `model._BACKBONES` 등록 + `validate_config._VALID_BACKBONES` + `tasks.py` 메타 |
| 새로운 태스크 | `utils.get_task_params` + `dataset_downloader.build_*_manifest` + `_VALID_TASKS` + `tasks.py` |
| 새로운 augmentation | `data_processor._augment_*` |
| 새로운 loss | `trainer.build_loss` 분기 + `_VALID_LOSSES` 추가 |
| 새로운 메트릭 | `trainer.build_metrics`, `evaluator` 의 계산 블록 |
| 새 API endpoint | `backend/api/routes/<name>.py` + `app.py` include_router |
| 새 페이지/UI | `frontend/src/pages/<name>.tsx` + `App.tsx` Route + `Layout.tsx` 네비 |
| 분산 학습 지원 | `trainer.GradAccumModel` 의 단일 GPU 가드 해제 + `tf.distribute` 호환 누적 구현 필요 |

## 13. 알려진 제약

- `GradAccumModel` 은 **단일 GPU 전용** (생성자에서 `num_replicas_in_sync > 1` 이면 `RuntimeError`).
- 인증/멀티유저 없음. 본 서비스는 **단일 사용자 MVP** 가정 — 운영 환경에 그대로 노출 금지.
- 부트스트랩 가중치는 합성 데이터(우폐 패치) 기반 — 임상적 의미 **0**. UI 동작 검증 용도만.
- NIH Pneumonia 라벨은 텍스트 마이닝 기반 weak label → AUROC 0.6 내외가 현실적 상한.
- ISIC GroundTruth 는 7-class만 사용 (SCC 제외, UNK 행 제거).
- DICOM/NIfTI 미지원 (예측은 `.png/.jpg/.jpeg`만).
- TensorFlow Addons 미설치 시 X-ray 회전 증강 자동 비활성 (런타임 1회 경고).
- CORS는 dev (`:5173`) 한정. 운영 시 frontend `dist/` 를 backend `StaticFiles` 로 동일 출처 제공 권장.
