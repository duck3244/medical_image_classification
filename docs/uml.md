# UML Diagrams

> Mermaid 기반 다이어그램. GitHub/IDE 미리보기에서 그대로 렌더링된다.

## 1. Component Diagram

```mermaid
flowchart TB
    CLI["main.py<br/>(argparse)"]

    subgraph Orchestration
        Bootstrap["utils.bootstrap()"]
        Config["utils.load_config()<br/>+ validate_config()"]
    end

    subgraph UseCases["Use-case Layer"]
        Trainer["trainer.train()"]
        Evaluator["evaluator.evaluate()"]
        Predictor["predictor.predict()"]
        Downloader["dataset_downloader.run()"]
    end

    subgraph Domain
        Model["model.py<br/>build_model / unfreeze"]
        Loss["trainer<br/>focal_loss / GradAccumModel"]
    end

    subgraph Data
        DataProc["data_processor.py<br/>build_pipeline / build_datasets"]
        Manifest[("manifest CSV<br/>train/val/test")]
    end

    subgraph Infra
        Utils["utils.py<br/>seed / GPU / logger / save_json"]
    end

    CLI --> Bootstrap
    CLI --> Config
    CLI --> Trainer
    CLI --> Evaluator
    CLI --> Predictor
    CLI --> Downloader

    Trainer --> Model
    Trainer --> Loss
    Trainer --> DataProc
    Trainer --> Utils

    Evaluator --> Model
    Evaluator --> DataProc

    Predictor --> Model
    Predictor --> Utils

    Downloader --> Manifest
    DataProc --> Manifest
    DataProc --> Model

    Bootstrap --> Utils
    Config --> Utils
```

## 2. Class Diagram

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

    GradAccumModel --|> kerasModel : extends
    GradAccumFlushCallback --|> kerasCallback : extends
    InnerWeightsCheckpoint --|> kerasCallback : extends
    GradAccumFlushCallback ..> GradAccumModel : flushes
    InnerWeightsCheckpoint ..> kerasModel : saves inner weights

    Config ..> TaskParams : derives via get_task_params()
    Predictor ..> BackboneRegistry : build_model + load_weights
    DataPipeline ..> BackboneRegistry : get_preprocess_fn

    class kerasModel {
        <<external>>
        tf.keras.Model
    }
    class kerasCallback {
        <<external>>
        tf.keras.callbacks.Callback
    }
```

## 3. Sequence — Training (`--mode train`)

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

## 4. Sequence — Inference + Grad-CAM (`--mode predict`)

```mermaid
sequenceDiagram
    actor User
    participant CLI as main.py
    participant P as predictor.predict
    participant M as model.build_model
    participant GC as grad_cam
    participant FS as filesystem

    User->>CLI: python main.py --mode predict --image x.png --weights best.h5
    CLI->>P: predict(cfg, weights, image, ...)
    P->>M: build_model(backbone, num_classes, img_size)
    M-->>P: keras.Model
    P->>FS: load_weights(weights_path)

    P->>P: _load_and_prepare(image)
    Note right of P: cv2 read → CLAHE? → resize → backbone preprocess
    P->>P: model.predict_on_batch(x)

    P->>GC: grad_cam(model, x, class_idx?)
    GC->>GC: _find_last_conv_layer() [Conv2D 타입 검사]
    GC->>GC: GradientTape → pooled grads → ReLU → normalize
    GC-->>P: heatmap (H,W) [0..1]

    P->>P: overlay_heatmap + watermark (Disclaimer)
    P->>FS: save <stem>_gradcam.png + result.json
    P-->>CLI: result dict
    CLI-->>User: 로그 + 산출물 경로
```

## 5. Sequence — Dataset Download (ISIC)

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

## 6. State — Stage Transition (Trainer)

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

## 7. Activity — Data Pipeline (한 샘플 처리)

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

## 8. Deployment View

```mermaid
flowchart LR
    subgraph Workstation["Single Workstation (RTX 4060 8GB)"]
        Code["Python 3.9<br/>TF 2.x"]
        GPU["NVIDIA RTX 4060 Laptop GPU<br/>~5.3GB usable VRAM"]
        Disk["Local disk<br/>~50GB free"]
    end

    subgraph Datasets
        HF["HuggingFace Hub<br/>NIH-Chest-X-ray-dataset"]
        S3["ISIC challenge S3<br/>isic-challenge-data"]
        Box["NIH Box (legacy)"]
    end

    Code -->|tf.config.set_memory_growth| GPU
    Code -->|hf_hub_download| HF
    Code -->|requests stream| S3
    Code -. fallback .-> Box
    Code -->|read/write| Disk
    GPU -->|VRAM| Code
```
