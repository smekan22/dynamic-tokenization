#!/bin/bash
#SBATCH -J uner_adapter_word_tk_256r
#SBATCH --time=5:0:0
#SBATCH --gres=gpu:1
#SBATCH --nodelist=ltl-gpu04

train() {
  local pattern=$1
  local out_prefix=$2
  local method=$3
  local dropout=$4
  local rank=$5
  local seed=$6
  local tokenization_type=$7
  local dynamic_tokenization_merges=$8
  local exp_type=$9
  local lng="${10}"
  local prefix="${11}"
  local alpha=$((2 * rank))

  mkdir -p $out_prefix

  today=$(date +%d%m%Y)

  local extra_args=""
  if [[ "$method" == "adalora" ]]; then
    extra_args="--adalora True --lora_alpha $alpha --lora_dropout $dropout --lora_rank $rank --tokenization_type $tokenization_type --dynamic_tokenization_merges $dynamic_tokenization_merges"
  else
    extra_args="--lora True --lora_alpha $alpha --lora_dropout $dropout --lora_rank $rank --tokenization_type $tokenization_type --dynamic_tokenization_merges $dynamic_tokenization_merges"
  fi

  python encoders/training/uner/run_ner_dynamic.py \
    --model_name_or_path $(printf $pattern "en") \
    --push_to_hub False \
    --dataset_name="universalner/universal_ner" \
    --dataset_config_name="en_ewt" \
    --do_train \
    --do_eval \
    --seed $seed \
    --exp_type $exp_type \
    --log_level=info \
    --evaluation_strategy=epoch \
    --save_strategy=epoch \
    --max_seq_length 128 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --learning_rate 1e-4 \
    --num_train_epochs 20.0 \
    --output_dir $out_prefix/${prefix}_${exp_type}_${lng}_${method}_alpha_${alpha}_drop_${dropout}_rank_${rank}_seed_${seed}_merges_${dynamic_tokenization_merges}_${today} \
    --overwrite_output_dir \
    --bf16 \
    --train_adapter \
    --report_to="wandb" \
    $extra_args

  return 0
}

source /mnt/nas_home/dmf45/miniconda3/bin/activate base

train "xlm-roberta-base" "output_lora_peft/ner/dynamic_tk" "lora" 0.3 256 42 "dynamic" 36 "dynamic_bpe" "en" "UNER_75PERC_"