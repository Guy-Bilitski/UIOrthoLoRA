import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Fine-tune a PEFT model with a subset of the dataset.")

    parser.add_argument("--model_id", type=str, required=True,
                        help="Hugging Face model ID (e.g., meta-llama/Llama-3.2-3B)")
    
    parser.add_argument("--peft_type", type=str, required=True,
                        choices=["lora", "vera", "randlora", "uiortholora"],
                        help="Type of PEFT method")
    
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save fine-tuned model")
    
    parser.add_argument("--num_epochs", type=int, default=1,
                        help="Number of training epochs (default: 1)")
    
    parser.add_argument("--learning_rate", type=float, default=1e-2,
                        help="Learning rate (default: 1e-2)")
    
    parser.add_argument("--alpha", type=int, default=None,
                        help="Alpha parameter (required for LoRA only)")

    parser.add_argument("--dropout", type=float, default=0.0,
                        help="Dropout for PEFT (default: 0.0)")
    
    parser.add_argument("--training_number", type=int, required=True,
                        help="Number of training examples to use (between 0 and total examples)")

    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    
    parser.add_argument("--results_path", type=str, required=True,
                        help="Path to the results file containing the dataset")
    
    parser.add_argument("--sc_number", type=int, default=5,
                        help="Number of self-consistency generations (default: 5)")
    
    parser.add_argument("--include_training", default=False, action='store_true')

    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to the pre-trained model (if different from model_id)")

    parser.add_argument("--lora_rank", type=int,
                        help="Rank for LoRA")

    parser.add_argument("--vera_rank", type=int,
                        help="Rank for VERA")

    parser.add_argument("--rand_lora_rank", type=int,
                        help="Rank for RandLORA")

    parser.add_argument("--svalues", type=int,
                        help="Number of singular values to adapt for UIOrthoLORA")

    parser.add_argument("--svectors", type=int,
                        help="Number of singular vectors to adapt for UIOrthoLORA")

    args = parser.parse_args()
    
    if args.include_training:
        if args.model_path is None:
            parser.error("--model_path is required when --include_training is True")

    return args
