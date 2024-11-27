#!/bin/bash
#SBATCH -J xnli_adapters_lora
#SBATCH --time=20:0:0
#SBATCH --gres=gpu:1
#SBATCH --nodelist=ltl-gpu04

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

  python encoders/training/xnli/run_glue.py \
    --model_name_or_path $(printf $pattern "en") \
    --push_to_hub True \
    --dataset_name=xnli \
    --dataset_config_name=en \
    --do_train \
    --do_eval \
    --seed $seed \
    --log_level=info \
    --evaluation_strategy=epoch \
    --save_strategy=epoch \
    --max_seq_length 128 \
    --per_device_train_batch_size 32 \
    --learning_rate 3e-4 \
    --num_train_epochs 15.0 \
    --output_dir $out_prefix/xnli_en_${method}_alpha_${alpha}_drop_${dropout}_rank_${rank}_seed_${seed} \
    --overwrite_output_dir \
    --bf16 \
    --train_adapter \
    --report_to="wandb" \
    $extra_args

  languages=('en')

  source /mnt/nas_home/dmf45/miniconda3/bin/activate base

  exp_types=("plain" "fvt" "original_tk_hypernet" "word_tk_hypernet")

  for lng in "${languages[@]}"; do
    for exp_type in "${exp_types[@]}"; do
      python encoders/evaluation/xnli/xnli_evaluation_peft_adapters.py --exp_type $exp_type --lng $lng --adapter_path $out_prefix/xnli_en_${method}_alpha_${alpha}_drop_${dropout}_rank_${rank}_seed_${seed}
    done
  done
}

patterns=("xlm-roberta-base")
out_prefixes=("output_lora_peft")
methods=("adalora")
dropouts=(0.3)
ranks=(32)
seeds=(42 123 456)

for pattern in "${patterns[@]}"; do
  for out_prefix in "${out_prefixes[@]}"; do
    for method in "${methods[@]}"; do
      for dropout in "${dropouts[@]}"; do
        for rank in "${ranks[@]}"; do
          for seed in "${seeds[@]}"; do
            train $pattern $out_prefix $method $dropout $rank $seed
          done
        done
      done
    done
  done
done
