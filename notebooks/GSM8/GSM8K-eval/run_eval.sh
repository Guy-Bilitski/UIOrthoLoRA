# python main.py \
#   --model_name_or_path mistralai/Mistral-7B-v0.1 \
#   --output_dir ./results/mistral-uiortholora-64-64-lr2e-3 \
#   --seed 42 \

python main.py \
--model_name_or_path mistralai/Mistral-7B-v0.1 \
--adapter_path ../uiortholora-mistral-run-64-64-lr2e-3/checkpoint-6250/adapter_model \
--output_dir ./results/mistral-uiortholora-64-64-lr2e-3 \
--seed 42 \
