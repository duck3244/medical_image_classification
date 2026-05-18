# UML Diagrams

> Mermaid 기반 다이어그램. GitHub/IDE 미리보기에서 그대로 렌더링된다.
> 풀스택(React ↔ FastAPI ↔ TF 파이프라인) 전체를 다룬다.

## 1. Component Diagram (Full Stack)

```mermaid
flowchart TB
    Browser["Browser (single user)"]

    subgraph Frontend["frontend/ (React 18 + Vite 5)"]
        App["App.tsx<br/>Routes"]
        Layout["Layout.tsx<br/>+ HealthDot (polling 15s)"]
        Predict["pages/PredictPage"]
        Models["pages/ModelsPage"]
        Disclaimer["DisclaimerBanner<br/>DisclaimerModal"]
        Client["api/client.ts<br/>fetch wrappers"]
        QC["@tanstack/react-query<br/>QueryClient"]
    end

    subgraph Backend["backend/ (FastAPI)"]
        AppPy["app.py<br/>lifespan / CORS / StaticFiles"]
        RHealth["api/routes/health"]
        RTasks["api/routes/tasks"]
        RModels["api/routes/models"]
        RPredict["api/routes/predict"]
        SInf["services/inference<br/>cache + asyncio.Lock"]
        SBoot["services/bootstrap<br/>synthetic auto-train"]
    end

    subgraph CLI["CLI (backend/main.py)"]
        Main["main.py<br/>argparse"]
    end

    subgraph Pipeline["ML Pipeline"]
        Utils["utils.py<br/>seed/GPU/validate_config"]
        DataProc["data_processor.py<br/>tf.data"]
        Model["model.py<br/>backbone factory"]
        Trainer["trainer.py<br/>2-stage + GradAccum"]
        Eval["evaluator.py"]
        Predr["predictor.py<br/>+ Grad-CAM"]
        Downloader["dataset_downloader.py"]
    end

    FS[("Filesystem<br/>dataset/ models/<br/>uploads/ results/")]

    Browser --> App
    App --> Layout
    Layout --> Predict
    Layout --> Models
    Layout --> Disclaimer
    Predict --> Client
    Models --> Client
    Layout --> Client
    Client --> QC

    Client -- "/api/* /static/*" --> AppPy
    AppPy --> RHealth
    AppPy --> RTasks
    AppPy --> RModels
    AppPy --> RPredict

    AppPy --> SInf
    AppPy --> SBoot
    RPredict --> SInf
    RHealth --> SInf
    SInf --> Model
    SInf --> Predr
    SBoot --> Trainer
    SBoot --> FS

    Main --> Utils
    Main --> Trainer
    Main --> Eval
    Main --> Predr
    Main --> Downloader

    Trainer --> Model
    Trainer --> DataProc
    Eval --> Model
    Eval --> DataProc
    Predr --> Model
    DataProc --> FS
    Downloader --> FS
    AppPy -. "/static/results" .-> FS
```

## 2. Class Diagram (Backend)

```mermaid
classDiagram
    class Config {
        +task: str
        +backbone: str
        +batch_size: int
        +grad_accum_steps: int
        +lr_head: float
        +lr_finetune: float
        +loss: str
        +focal_alpha: float
        +focal_gamma: float
        +paths: dict
        +validate() void
    }

    class TaskParams {
        +task: str
        +img_size: int
        +num_classes: int
        +class_names: List~str~
        +is_binary: bool
        +data_dir: str
    }

    class BackboneRegistry {
        -_BACKBONES: dict
        +get_preprocess_fn(backbone) Callable
        +build_model(backbone, num_classes, img_size) Tuple~Model, Model~
        +unfreeze_for_finetune(base, fine_tune_at, freeze_bn) dict
    }

    class GradAccumModel {
        -inner: keras.Model
        -accum_steps: int
        -_accum_grads: List~Variable~
        -_accum_counter: Variable
        +call(inputs, training)
        +train_step(data) dict
        +test_step(data) dict
        +flush_accumulated() void
        +reset_accumulators() void
    }

    class GradAccumFlushCallback {
        -_accum_model: GradAccumModel
        +on_epoch_end(epoch, logs) void
    }

    class InnerWeightsCheckpoint {
        -inner: keras.Model
        -filepath: str
        -monitor: str
        -mode: str
        -best: float
        +on_epoch_end(epoch, logs) void
    }

    class FocalLoss {
        <<factory>>
        +binary_focal_loss(alpha, gamma) Callable
        +categorical_focal_loss(alpha, gamma) Callable
        +build_loss(is_binary, kind, alpha, gamma) Callable
    }

    class DataPipeline {
        <<module>>
        +load_manifest(csv_path) DataFrame
        +compute_class_weights(labels) dict
        +build_pipeline(manifest, ...) tf.data.Dataset
        +build_datasets(data_dir, ...) Tuple
    }

    class DatasetDownloader {
        <<module>>
        +download_nih(nih_dir, subset, source) bool
        +download_isic(isic_dir) bool
        +build_nih_manifest(nih_dir) void
        +build_isic_manifest(isic_dir) void
        -_isic_lesion_group_split(df) Tuple
    }

    class Predictor {
        <<module>>
        +load_model_for_inference(config, weights) Model
        +predict_one(config, model, image_path, out_dir) dict
        +grad_cam(model, image, class_idx) ndarray
        -_find_last_conv_layer(model) str
    }

    class InferenceService {
        <<module>>
        -_lock: asyncio.Lock
        -_model: keras.Model
        -_model_key: Tuple~str,str,str~
        +ensure_model(cfg, weights) Model
        +warmup(cfg, weights) bool
        +run_predict_one(cfg, weights, image, out_dir) dict
        +find_default_weights(models_dir) Path
        -_set_fp32_policy_if_mixed() void
    }

    class BootstrapService {
        <<module>>
        +generate_synthetic_dataset(data_dir, logger) void
        +run_bootstrap_training(cfg, backend_dir, logger) Path
        +auto_bootstrap(cfg, backend_dir, models_dir, logger) Path
    }

    class FastAPIApp {
        <<entry>>
        +app: FastAPI
        +lifespan(app) async
        -_make_logger() Logger
    }

    GradAccumModel --|> kerasModel : extends
    GradAccumFlushCallback --|> kerasCallback : extends
    InnerWeightsCheckpoint --|> kerasCallback : extends
    GradAccumFlushCallback ..> GradAccumModel : flushes
    InnerWeightsCheckpoint ..> kerasModel : saves inner weights

    Config ..> TaskParams : derives via get_task_params()
    Predictor ..> BackboneRegistry : build_model + load_weights
    DataPipeline ..> BackboneRegistry : get_preprocess_fn

    FastAPIApp ..> InferenceService : warmup
    FastAPIApp ..> BootstrapService : auto_bootstrap
    InferenceService ..> Predictor : predict_one
    InferenceService ..> BackboneRegistry : build_model
    BootstrapService ..> DataPipeline : (via trainer.train)

    class kerasModel {
        <<external>>
        tf.keras.Model
    }
    class kerasCallback {
        <<external>>
        tf.keras.callbacks.Callback
    }
```

## 3. Class Diagram (Frontend, simplified)

```mermaid
classDiagram
    class App {
        <<component>>
        +Routes
    }

    class Layout {
        <<component>>
        +HealthDot (useQuery health)
        +DisclaimerBanner
        +DisclaimerModal
        +Outlet
    }

    class PredictPage {
        <<component>>
        -file: File
        -previewUrl: string
        -task / backbone / weights: string
        +useQuery(tasks)
        +useQuery(models)
        +useMutation(postPredict)
    }

    class ModelsPage {
        <<component>>
        +useQuery(models)
    }

    class ApiClient {
        <<module>>
        +fetchHealth() HealthResponse
        +fetchTasks() TasksResponse
        +fetchModels() ModelsResponse
        +postPredict(input) PredictResponse
    }

    class Types {
        <<module>>
        +HealthResponse
        +TasksResponse
        +ModelsResponse
        +PredictResponse
        +PredictResult
        +ModelRun / WeightsFile
    }

    App --> Layout
    Layout --> PredictPage
    Layout --> ModelsPage
    Layout --> ApiClient : fetchHealth (polling)
    PredictPage --> ApiClient
    ModelsPage --> ApiClient
    ApiClient ..> Types
```

## 4. Sequence — Server Startup (`uvicorn app:app`)

```mermaid
sequenceDiagram
    actor Op as Operator
    participant U as uvicorn
    participant App as app.py (lifespan)
    participant Cfg as utils.load_config
    participant GPU as utils.setup_gpu_memory_growth
    participant Boot as services.bootstrap.auto_bootstrap
    participant Inf as services.inference
    participant FS as filesystem

    Op->>U: uvicorn app:app --port 8000
    U->>App: enter lifespan
    App->>Cfg: load_config(backend/config.json)
    Cfg->>Cfg: validate_config()
    Cfg-->>App: cfg
    App->>GPU: set memory_growth on all GPUs
    App->>Inf: find_default_weights(models_dir)
    alt 가중치 없음
        App->>Boot: auto_bootstrap(cfg, backend_dir, models_dir)
        Boot->>FS: generate synthetic dataset (60 imgs)
        Boot->>Boot: trainer.train(boot_cfg) [1+1 epoch]
        Boot-->>App: best.weights.h5 path
    else 가중치 있음
        Inf-->>App: existing path
    end
    App->>Inf: warmup(cfg, weights)
    Inf->>Inf: _set_fp32_policy_if_mixed
    Inf->>Inf: build_model + load_weights (asyncio.to_thread)
    Inf-->>App: OK
    App-->>U: ready
    Op->>U: HTTP traffic
```

## 5. Sequence — Predict (POST /api/predict)

```mermaid
sequenceDiagram
    actor User
    participant FE as PredictPage (React)
    participant Q as react-query
    participant API as FastAPI /api/predict
    participant Inf as services.inference
    participant P as predictor.predict_one
    participant GC as grad_cam
    participant FS as filesystem

    User->>FE: choose file + (optional) overrides → Submit
    FE->>Q: useMutation(postPredict)
    Q->>API: POST multipart/form-data
    API->>API: _validate_upload (ext / size ≤ 10MB)
    API->>API: _build_request_config (override merge)
    API->>API: get_task_params(cfg)  # num_classes/img_size 검증
    API->>FS: write uploads/<uuid>.<ext>
    API->>Inf: run_predict_one(cfg, weights, image, out_dir)
    Inf->>Inf: ensure_model() [lock + lazy reload if key changed]
    Inf->>P: asyncio.to_thread(predict_one, ...)
    P->>P: _load_and_prepare (cv2 → CLAHE? → resize → preprocess)
    P->>P: model.predict_on_batch(x)
    P->>GC: grad_cam (Conv2D auto-detect)
    GC-->>P: heatmap
    P->>FS: write results/predict_<ts>_<hex>/<stem>_gradcam.png
    P-->>Inf: result dict
    Inf-->>API: result
    API->>API: relative path → "/static/results/..."
    API-->>Q: PredictResponse (disclaimer + result + gradcam_url)
    Q-->>FE: data
    FE->>FE: render class probs + Grad-CAM <img src>
    FE->>API: GET /static/results/<run>/<file>  (img tag)
    API->>FS: serve PNG
```

## 6. Sequence — Training CLI (`--mode train`)

```mermaid
sequenceDiagram
    actor User
    participant CLI as main.py
    participant U as utils
    participant T as trainer.train
    participant DP as data_processor
    participant M as model
    participant GA as GradAccumModel
    participant FS as filesystem

    User->>CLI: python main.py --mode train
    CLI->>U: load_config(config.json)
    U->>U: validate_config()
    U-->>CLI: cfg
    CLI->>U: bootstrap(cfg)
    U->>U: set_seed / GPU / mixed_precision / XLA
    U-->>CLI: logger
    CLI->>T: train(cfg, logger)

    T->>DP: build_datasets(...)
    DP->>FS: read train.csv / val.csv / test.csv
    DP-->>T: train_ds, val_ds, test_ds, meta(class_weights)

    T->>M: build_model(backbone, num_classes, img_size)
    M-->>T: (model, base_model)  [base frozen]

    alt accum_steps > 1
        T->>GA: wrap(inner=model, accum_steps)
    end

    T->>T: build_loss + build_metrics + compile (lr_head)
    T->>T: build_callbacks(stage1)

    rect rgb(240, 248, 255)
        Note over T: Stage 1 — head only
        T->>GA: fit(train_ds, val_ds, epochs_head)
        loop each batch
            GA->>GA: train_step → accumulate grads → tf.cond apply
        end
        GA-->>T: hist1
    end

    T->>M: unfreeze_for_finetune(base, fine_tune_at, freeze_bn)
    T->>GA: reset_accumulators()
    T->>T: re-compile (lr_finetune)
    T->>T: build_callbacks(stage2)

    rect rgb(245, 245, 220)
        Note over T: Stage 2 — fine-tune
        T->>GA: fit(train_ds, val_ds, epochs_finetune)
        GA-->>T: hist2
    end

    T->>FS: save final.weights.h5 + history.json
    T->>T: best_path = stage2/best ?? stage1/best ?? final
    alt 모든 경로 부재
        T-->>CLI: raise FileNotFoundError
    end
    T-->>CLI: {run_dir, best_weights, history, meta}
    CLI-->>User: log "학습 완료"
```

## 7. Sequence — Dataset Download (ISIC)

```mermaid
sequenceDiagram
    actor User
    participant CLI as main.py
    participant DD as dataset_downloader
    participant S3 as ISIC S3
    participant FS as filesystem

    User->>CLI: --mode download --task isic_multiclass --agree
    CLI->>DD: download_isic(isic_dir, agreed=True)
    DD->>DD: show_license_and_require_agree()
    loop ISIC_URLS (Input.zip / GroundTruth.csv / Metadata.csv)
        DD->>S3: GET (resumable, tqdm)
        S3-->>DD: bytes
        DD->>FS: write raw/<file>
    end
    DD->>FS: unzip Training_Input.zip → images/
    DD->>DD: build_isic_manifest()
    alt Metadata.csv 존재 & lesion_id 컬럼 있음
        DD->>DD: _isic_lesion_group_split (StratifiedGroupKFold)
        Note right of DD: lesion-level grouping<br/>누수 검증 후 raise on overlap
    else 메타데이터 부재
        DD->>DD: image-level stratified split + warning
    end
    DD->>FS: write train.csv / val.csv / test.csv
    DD-->>CLI: True
```

## 8. State — Inference Model Cache

```mermaid
stateDiagram-v2
    [*] --> Empty
    Empty --> Loading : ensure_model(key=k)
    Loading --> Loaded : build_model + load_weights OK
    Loading --> Empty : exception (no retry, propagate)
    Loaded --> Serving : run_predict_one (lock)
    Serving --> Loaded : predict_one done
    Loaded --> Reloading : ensure_model(key≠current)
    Reloading --> Loaded : new model installed
    Loaded --> [*] : process exit
```

## 9. State — Trainer Stage Transition

```mermaid
stateDiagram-v2
    [*] --> BuildModel
    BuildModel --> Stage1Compile : base.trainable=False
    Stage1Compile --> Stage1Fit : optimizer(lr_head)
    Stage1Fit --> Stage1Fit : epoch end → flush_accum
    Stage1Fit --> Unfreeze : EarlyStopping or epochs_head 도달
    Unfreeze --> Stage2Compile : unfreeze_for_finetune(fine_tune_at, freeze_bn)
    Stage2Compile --> Stage2Fit : reset_accumulators + re-compile(lr_finetune)
    Stage2Fit --> Stage2Fit : epoch end → flush_accum
    Stage2Fit --> Saving : EarlyStopping or epochs_finetune 도달
    Saving --> ResolveBest : save final.weights.h5
    ResolveBest --> [*] : stage2/best ✓
    ResolveBest --> [*] : stage1/best ✓ (fallback)
    ResolveBest --> [*] : final ✓ (warn fallback)
    ResolveBest --> Error : 모두 부재 → FileNotFoundError
    Error --> [*]
```

## 10. Activity — Data Pipeline (한 샘플 처리)

```mermaid
flowchart LR
    Start([path, label]) --> Decode["tf.io.decode_image<br/>(uint8, channels=3)"]
    Decode --> XrayCheck{task ==<br/>nih_cxr_binary?}
    XrayCheck -- yes --> CLAHE["py_function: CLAHE<br/>gray → eq → 3ch"]
    XrayCheck -- no --> Resize
    CLAHE --> Resize["resize(img_size)<br/>→ float32"]
    Resize --> AugCheck{augment?}
    AugCheck -- yes & xray --> AugX["brightness/contrast<br/>zoom/translate<br/>+ rotate (tfa)"]
    AugCheck -- yes & skin --> AugS["flip H/V<br/>brightness/contrast/sat"]
    AugCheck -- no --> Pre
    AugX --> Pre
    AugS --> Pre
    Pre["preprocess_input<br/>(backbone-specific)"]
    Pre --> Label{is_binary?}
    Label -- yes --> Sigmoid["label → float32 [1]"]
    Label -- no --> OneHot["one_hot(num_classes)"]
    Sigmoid --> Out([x, y])
    OneHot --> Out
```

## 11. Activity — Predict Request End-to-End

```mermaid
flowchart TB
    UStart([User picks file in PredictPage]) --> Submit{Submit clicked}
    Submit --> Mutation["useMutation(postPredict)"]
    Mutation --> Multipart["FormData: file + overrides"]
    Multipart --> Fetch["fetch('/api/predict', POST)"]
    Fetch --> Validate{ext/size OK?}
    Validate -- no --> Err400["400 / 413 with detail"]
    Validate -- yes --> CfgMerge["build cfg (task/backbone override)"]
    CfgMerge --> TP{"get_task_params(cfg)"}
    TP -- ValueError --> Err400
    TP -- OK --> WeightsPick{weights override or default?}
    WeightsPick -- none --> Err503["503 no weights"]
    WeightsPick -- ok --> Save["save uploads/<uuid>.<ext>"]
    Save --> Lock["asyncio.Lock + to_thread"]
    Lock --> Ensure["ensure_model(cache key)"]
    Ensure --> PredOne["predictor.predict_one"]
    PredOne --> GradCAM["overlay + watermark<br/>save PNG"]
    GradCAM --> URL["build /static/results URL"]
    URL --> Resp["JSON: disclaimer + result"]
    Resp --> Render["React: probs + img src"]
    Render --> UEnd([User sees result + Grad-CAM])
    Err400 --> Render
    Err503 --> Render
```

## 12. Deployment View

```mermaid
flowchart LR
    subgraph Workstation["Single Workstation (RTX 4060 8GB)"]
        Node["Node 18 + Vite dev (:5173)"]
        Py["Python 3.9 + TF 2.10 + FastAPI/uvicorn (:8000)"]
        GPU["NVIDIA RTX 4060 Laptop GPU<br/>~5.3GB usable VRAM"]
        Disk["Local disk<br/>~50GB free"]
    end

    subgraph Datasets
        HF["HuggingFace Hub<br/>NIH-Chest-X-ray-dataset"]
        S3["ISIC challenge S3<br/>isic-challenge-data"]
        Box["NIH Box (legacy)"]
    end

    Br["Browser"] -->|http :5173| Node
    Node -->|proxy /api /static| Py
    Py -->|tf.config.set_memory_growth| GPU
    Py -->|hf_hub_download| HF
    Py -->|requests stream| S3
    Py -. fallback .-> Box
    Py -->|read/write| Disk
    GPU -->|VRAM| Py
```
