#!/bin/bash
#SBATCH -J xnli_adapter_sampling_per_batch
#SBATCH --time=20:0:0
#SBATCH --gres=gpu:1
#SBATCH --nodelist=ltl-gpu04

train_uniform() {
  local pattern=$1
  local out_prefix=$2
  local method=$3
  local dropout=$4
  local rank=$5
  local seed=$6
  local tokenization_type=$7
  local exp_type=$8
  local lng=$9
  local lr=${10}
  local epochs=${11}
  local dynamic_tokenization_merges=0
  local alpha=$((2 * rank))

  mkdir -p $out_prefix

  today=$(date +%d%m%Y)

  local extra_args=""
  if [[ "$method" == "adalora" ]]; then
    extra_args="--adalora True --lora_alpha $alpha --lora_dropout $dropout --lora_rank $rank --tokenization_type $tokenization_type --dynamic_tokenization_merges $dynamic_tokenization_merges"
  else
    extra_args="--lora True --lora_alpha $alpha --lora_dropout $dropout --lora_rank $rank --tokenization_type $tokenization_type --dynamic_tokenization_merges $dynamic_tokenization_merges"
  fi

  python encoders/training/xnli/run_glue_dynamic.py \
    --model_name_or_path $(printf $pattern "en") \
    --push_to_hub False \
    --dataset_name=xnli \
    --dataset_config_name=en \
    --do_train \
    --do_eval \
    --seed $seed \
    --exp_type $exp_type \
    --do_tokeniser_sampling_per_batch \
    --log_level=info \
    --evaluation_strategy=steps \
    --eval_steps 5000 \
    --save_strategy=steps \
    --save_steps 5000 \
    --max_seq_length 128 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --learning_rate $lr \
    --num_train_epochs $epochs \
    --output_dir $out_prefix/UNIFORM_TOKENISER_SAMPLING_per_batch_${exp_type}_${lng}_${method}_alpha_${alpha}_drop_${dropout}_rank_${rank}_seed_${seed}_merges_${dynamic_tokenization_merges}_${today}_RETRY_${lr}_${epochs} \
    --overwrite_output_dir \
    --bf16 \
    --train_adapter \
    --report_to="wandb" \
    $extra_args

  return 0
}

train_gaussian() {
  local pattern=$1
  local out_prefix=$2
  local method=$3
  local dropout=$4
  local rank=$5
  local seed=$6
  local tokenization_type=$7
  local exp_type=$8
  local lng=$9
  local lr=${10}
  local epochs=${11}
  local dynamic_tokenization_merges=0
  local alpha=$((1 * rank))

  mkdir -p $out_prefix

  today=$(date +%d%m%Y)

  local extra_args=""
  if [[ "$method" == "adalora" ]]; then
    extra_args="--adalora True --lora_alpha $alpha --lora_dropout $dropout --lora_rank $rank --tokenization_type $tokenization_type --dynamic_tokenization_merges $dynamic_tokenization_merges"
  else
    extra_args="--lora True --lora_alpha $alpha --lora_dropout $dropout --lora_rank $rank --tokenization_type $tokenization_type --dynamic_tokenization_merges $dynamic_tokenization_merges"
  fi

  python encoders/training/xnli/run_glue_dynamic.py \
    --model_name_or_path $(printf $pattern "en") \
    --push_to_hub False \
    --dataset_name=xnli \
    --dataset_config_name=en \
    --do_train \
    --do_eval \
    --seed $seed \
    --exp_type $exp_type \
    --do_tokeniser_sampling_per_batch_gaussian \
    --log_level=info \
    --evaluation_strategy=steps \
    --eval_steps 5000 \
    --save_strategy=steps \
    --save_steps 5000 \
    --max_seq_length 128 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --learning_rate $lr \
    --num_train_epochs $epochs \
    --output_dir $out_prefix/GAUSSIAN_TOKENISER_SAMPLING_per_batch_${exp_type}_${lng}_${method}_alpha_${alpha}_drop_${dropout}_rank_${rank}_seed_${seed}_merges_${dynamic_tokenization_merges}_${today}_${lr}_${epochs} \
    --overwrite_output_dir \
    --bf16 \
    --train_adapter \
    --report_to="wandb" \
    $extra_args

  return 0
}

source /mnt/nas_home/dmf45/miniconda3/bin/activate base

train "xlm-roberta-base" "output_lora_peft/dynamic_tk" "lora" 0.3 128 42 "dynamic" "dynamic_bpe" "en" 1e-4 15.0
train_gaussian "xlm-roberta-base" "output_lora_peft/dynamic_tk" "lora" 0.3 128 42 "dynamic" "dynamic_bpe" "en"
