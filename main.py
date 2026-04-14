"""
main.py
medical_image_classification CLI 진입점.

⚠️ 본 프로젝트는 개인 학습·연구 목적이며, 실제 의료 진단에 사용할 수 없습니다.

사용 예시:
    python main.py --mode download --task nih_cxr_binary --agree
    python main.py --mode download --task nih_cxr_binary --agree --subset 0.1
    python main.py --mode download --task isic_multiclass --agree

    python main.py --mode train --task nih_cxr_binary --backbone EfficientNetB0
    python main.py --mode train --task isic_multiclass --backbone EfficientNetB0

    python main.py --mode eval  --task nih_cxr_binary --weights models/.../best.h5
    python main.py --mode predict --image sample.png --weights models/.../best.h5
    python main.py --mode predict --input_dir samples/  --weights models/.../best.h5
"""

import argparse
import sys

from utils import bootstrap, load_config


def parse_args():
    p = argparse.ArgumentParser(
        description="Medical Image Classification (학습용 / NOT FOR DIAGNOSIS)"
    )
    p.add_argument("--mode", required=True,
                   choices=["download", "train", "eval", "predict"])
    p.add_argument("--config", default="config.json",
                   help="설정 파일 경로")
    p.add_argument("--task", default=None,
                   choices=["nih_cxr_binary", "isic_multiclass"],
                   help="config의 task를 덮어씀")
    p.add_argument("--backbone", default=None,
                   choices=["EfficientNetB0", "MobileNetV2", "ResNet50"],
                   help="config의 backbone을 덮어씀")

    # download
    p.add_argument("--agree", action="store_true",
                   help="라이선스 동의 (다운로드에 필요)")
    p.add_argument("--subset", type=float, default=1.0,
                   help="NIH 다운로드 비율 (0<subset<=1.0)")
    p.add_argument("--source", default="huggingface",
                   choices=["huggingface", "box", "manual"],
                   help="NIH 데이터 소스 (기본: huggingface)")

    # eval / predict
    p.add_argument("--weights", default=None,
                   help="학습된 가중치 경로 (.h5)")
    p.add_argument("--image", default=None,
                   help="단일 이미지 추론 경로")
    p.add_argument("--input_dir", default=None,
                   help="배치 추론용 이미지 디렉토리")
    return p.parse_args()


def apply_overrides(config: dict, args) -> dict:
    if args.task:
        config["task"] = args.task
    if args.backbone:
        config["backbone"] = args.backbone
    return config


def cmd_download(config, args, logger):
    import dataset_downloader
    ok = dataset_downloader.run(
        task=config["task"],
        config=config,
        agreed=args.agree,
        subset=args.subset,
        source=args.source,
    )
    if not ok:
        logger.error("다운로드 실패")
        sys.exit(1)


def cmd_train(config, args, logger):
    from trainer import train
    result = train(config, logger)
    logger.info(f"학습 완료. best={result['best_weights']}")
    logger.info(f"run_dir={result['run_dir']}")


def cmd_eval(config, args, logger):
    if not args.weights:
        logger.error("--weights 경로가 필요합니다.")
        sys.exit(2)
    from evaluator import evaluate
    out = evaluate(config, args.weights, logger)
    logger.info(f"평가 완료. out_dir={out['out_dir']}")


def cmd_predict(config, args, logger):
    if not args.weights:
        logger.error("--weights 경로가 필요합니다.")
        sys.exit(2)
    if not args.image and not args.input_dir:
        logger.error("--image 또는 --input_dir 중 하나가 필요합니다.")
        sys.exit(2)
    from predictor import predict
    out = predict(
        config=config,
        weights_path=args.weights,
        image=args.image,
        input_dir=args.input_dir,
        logger=logger,
    )
    logger.info(f"추론 완료. out_dir={out['out_dir']}")


def main():
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)
    logger = bootstrap(config, logger_name=f"medimg_{args.mode}")

    dispatch = {
        "download": cmd_download,
        "train": cmd_train,
        "eval": cmd_eval,
        "predict": cmd_predict,
    }
    dispatch[args.mode](config, args, logger)


if __name__ == "__main__":
    main()
