"""Reproducible hyper-parameter runner for UIOrthoLoRA on E2E NLG."""

import argparse
import itertools
import os
from pathlib import Path

# ───────────────────────── settings every run shares ───────────────────────── #
MODEL_TYPE = "gpt2-medium"
FINETUNE   = True   # change to False if you only want to evaluate
INFERENCE = True
EVALUATE = True
OUTPUT_DIR = "outputs"
MODELS_DIR = "models"
RESULTS_DIR = "results"

LORA_TARGET_MODULES = [
    "attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"
]
LORA_ALPHA = 1
LORA_DROPOUT = 0

BASE_TRAIN_ARGS = dict(
    eval_strategy="no",
    save_strategy="no",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    eval_accumulation_steps=2,
    lr_scheduler_type="linear",
    label_smoothing_factor=0.1,
    num_train_epochs=5,
    weight_decay=0.01,
    warmup_steps=500,
    logging_steps=50,
    save_total_limit=1,
    report_to="none",
)

INFERENCE_ARGS = {
    "num_beams": 10,
    "no_repeat_ngram_size": 4,
    "length_penalty": 0.9,
    "max_new_tokens": 64,
    "inference_batch_size": 16,
}

# ───────────── default grid to try ───────────── #
SEARCH_LRS = [5e-2, 7e-2]
NUM_SVALUES = [256]
NUM_SVECTORS = [0, 15, 30]
SEEDS = [17]
INIT_SIGMA = [0.1]
INIT_SCALER = [0.1]


def parse_float_list(value):
    return [float(item) for item in value.split(",") if item]


def parse_int_list(value):
    return [int(item) for item in value.split(",") if item]


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run reproducible UIOrthoLoRA E2E training/evaluation sweeps."
    )
    parser.add_argument("--model-type", default=MODEL_TYPE)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--models-dir", default=MODELS_DIR)
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--learning-rates", type=parse_float_list, default=SEARCH_LRS)
    parser.add_argument("--num-svalues", type=parse_int_list, default=NUM_SVALUES)
    parser.add_argument("--num-svectors", type=parse_int_list, default=NUM_SVECTORS)
    parser.add_argument("--seeds", type=parse_int_list, default=SEEDS)
    parser.add_argument("--init-sigma", type=parse_float_list, default=INIT_SIGMA)
    parser.add_argument("--init-scaler", type=parse_float_list, default=INIT_SCALER)
    parser.add_argument("--epochs", type=float, default=BASE_TRAIN_ARGS["num_train_epochs"])
    parser.add_argument("--train-batch-size", type=int, default=BASE_TRAIN_ARGS["per_device_train_batch_size"])
    parser.add_argument("--eval-batch-size", type=int, default=BASE_TRAIN_ARGS["per_device_eval_batch_size"])
    parser.add_argument("--warmup-steps", type=int, default=BASE_TRAIN_ARGS["warmup_steps"])
    parser.add_argument("--weight-decay", type=float, default=BASE_TRAIN_ARGS["weight_decay"])
    parser.add_argument("--label-smoothing", type=float, default=BASE_TRAIN_ARGS["label_smoothing_factor"])
    parser.add_argument("--run-prefix", default="lr")
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--no-finetune", action="store_true")
    parser.add_argument("--no-inference", action="store_true")
    parser.add_argument("--no-evaluate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-de", action="store_true", help="Disable D and E diagonal scalers (they are frozen to 1 and not trained)")
    return parser


def build_lora_config(num_svalues, num_svectors, init_sigma, init_scaler, use_de=True):
    from peft import UIOrthoLoRAConfig

    return UIOrthoLoRAConfig(
        target_modules=LORA_TARGET_MODULES,
        fan_in_fan_out=True,
        initial_scaler=init_scaler,
        initial_sigma=init_sigma,
        uiortholora_alpha=LORA_ALPHA,
        uiortholora_dropout=LORA_DROPOUT,
        num_svalues_to_adapt=num_svalues,
        num_svectors_to_adapt=num_svectors,
        use_de=use_de,
    )


def build_training_args(args, results_path, lr, seed):
    from transformers.training_args import TrainingArguments

    train_args = dict(BASE_TRAIN_ARGS)
    train_args.update(
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        label_smoothing_factor=args.label_smoothing,
    )
    return TrainingArguments(
        output_dir=results_path,
        learning_rate=lr,
        seed=seed,
        data_seed=seed,
        **train_args,
    )


# ───────────────────────── main loop ───────────────────── #
def main() -> None:
    args = build_parser().parse_args()
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    from E2e_training2 import train_and_evaluate

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_dir, args.models_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_dir, args.results_dir).mkdir(parents=True, exist_ok=True)

    grid = itertools.product(
        args.learning_rates,
        args.num_svalues,
        args.num_svectors,
        args.seeds,
        args.init_sigma,
        args.init_scaler,
    )

    for lr, num_svalues, num_svectors, seed, init_sigma, init_scaler in grid:
        use_de = not args.no_de
        run_tag = (
            f"{args.run_prefix}_{lr:g}_svalues_{num_svalues}_svectors_{num_svectors}"
            f"_seed_{seed}_init_sigma_{init_sigma:g}_init_scaler_{init_scaler:g}"
            + ("" if use_de else "_no_de")
        )
        model_path = f"{args.output_dir}/{args.models_dir}/{run_tag}"
        results_path = f"{args.output_dir}/{args.results_dir}/{run_tag}"
        peft_config = build_lora_config(num_svalues, num_svectors, init_sigma, init_scaler, use_de=use_de)
        Path(model_path).mkdir(parents=True, exist_ok=True)
        Path(results_path).mkdir(parents=True, exist_ok=True)

        train_args = build_training_args(args, results_path, lr, seed)
        metadata = {
            "model_type": args.model_type,
            "learning_rate": lr,
            "seed": seed,
            "training_args": train_args.to_dict(),
            "peft_config": peft_config.to_dict(),
            "inference_args": INFERENCE_ARGS,
        }

        print(f"\n▶️  Starting run {run_tag}")
        if args.dry_run:
            print(f"Dry run: model_path={model_path}")
            print(f"Dry run: results_path={results_path}")
            continue

        train_and_evaluate(
            output_dir=args.output_dir,
            models_dir=args.models_dir,
            results_dir=args.results_dir,
            model_type=args.model_type,
            training_args=train_args,
            finetune=not args.no_finetune,
            peft_config=peft_config,
            inference_args=INFERENCE_ARGS,
            run_tag=run_tag,
            inference=not args.no_inference,
            evaluate=not args.no_evaluate,
            seed=seed,
            metadata=metadata,
        )
        print(f"✅ Finished run {run_tag}")

if __name__ == "__main__":
    main()
