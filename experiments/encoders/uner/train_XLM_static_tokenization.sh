#!/bin/bash
#SBATCH -J xnli_adapters_lora_adalora
#SBATCH --time=60:0:0
#SBATCH --gres=gpu:1
#SBATCH --nodelist=ltl-gpu01

train() {
  local pattern=$1
  local out_prefix=$2
  local method=$3
  local dropout=$4
  local rank=$5
  local seed=$6
  local alpha=$((2 * rank))

  mkdir -p $out_prefix

  local extra_args=""
  if [[ "$method" == "adalora" ]]; then
    extra_args="--adalora True --lora_alpha $alpha --lora_dropout $dropout --lora_rank $rank"
  else
    extra_args="--lora True --lora_alpha $alpha --lora_dropout $dropout --lora_rank $rank"
  fi

  python encoders/training/uner/run_ner.py \
    --model_name_or_path $(printf $pattern "en") \
    --push_to_hub True \
    --dataset_name="universalner/universal_ner" \
    --dataset_config_name="en_ewt" \
    --do_train \
    --do_eval \
    --seed $seed \
    --log_level=info \
    --evaluation_strategy=epoch \
    --save_strategy=epoch \
    --max_seq_length 128 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --learning_rate 3e-5 \
    --num_train_epochs 20.0 \
    --output_dir $out_prefix/ner/UNER_subword_tk_en_${method}_alpha_${alpha}_drop_${dropout}_rank_${rank}_seed_${seed}_lr_3e-5 \
    --overwrite_output_dir \
    --bf16 \
    --train_adapter \
    --report_to="wandb" \
    $extra_args

  return 0
}

train "xlm-roberta-base" "output_lora_peft" "lora" 0.2 256 42