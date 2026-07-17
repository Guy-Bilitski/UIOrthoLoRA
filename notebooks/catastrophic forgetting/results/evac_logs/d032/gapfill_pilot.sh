#!/bin/bash
cd "/home/guyb/UIOrthoLoRA/notebooks/catastrophic forgetting" || exit 1
export HF_HOME=/scratch/hf_cache HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export CUDA_VISIBLE_DEVICES=0
echo "[gapfill] start $(date -u +%H:%M:%SZ) on CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
/home/guy/UIOrthoLoRA/.venv/bin/python train_cs.py --method lora --base_model meta-llama/Llama-2-7b-hf --data_path repro/LLM-Adapters/ft-training_set/commonsense_170k.json --cutoff_len 256 --num_epochs 3 --learning_rate 0.0002 --weight_decay 0.0 --batch_size 16 --micro_batch_size 16 --warmup_steps 100 --target_modules q_proj,k_proj,v_proj,up_proj,down_proj --dropout 0.05 --train_on_inputs 1 --max_samples 0 --use_dora 0 --corda 0 --corda_calib_size 256 --milora 0 --sclora 0 --sclora_beta 0.5 --sclora_calib_size 256 --lora_r 16 --lora_alpha 32 --clora_k 512 --clora_lambda 1.0 --k_val 256 --k_vec 128 --alpha 1.0 --use_de 1 --initial_scaler 0.1 --initial_sigma 0.1 --seed 45 --run_name lrsw_lora_r16_lr2e4_s45 && /home/guy/UIOrthoLoRA/.venv/bin/python eval_one_gpu.py --adapter /scratch/cf_models/lrsw_lora_r16_lr2e4_s45 --run_name lrsw_lora_r16_lr2e4_s45 --base_model meta-llama/Llama-2-7b-hf --adapt_task cs --ret_suite broad --ret_limit 0 --ret_max_gen 512
echo "[gapfill] exit rc=$? $(date -u +%H:%M:%SZ)"
