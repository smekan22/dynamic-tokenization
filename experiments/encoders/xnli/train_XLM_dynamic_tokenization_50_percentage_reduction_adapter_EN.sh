#!/bin/bash
#SBATCH -J xnli_adapter_word_tk_256r
#SBATCH --time=10:0:0
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
    --log_level=info \
    --evaluation_strategy=steps \
    --save_strategy=steps \
    --save_steps 5000 \
    --eval_steps 5000 \
    --max_seq_length 128 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --learning_rate 1e-4 \
    --num_train_epochs 15.0 \
    --output_dir $out_prefix/${exp_type}_${lng}_${method}_alpha_${alpha}_dropout_${dropout}_rank_${rank}_seed_${seed}_merges_${dynamic_tokenization_merges}_${today} \
    --overwrite_output_dir \
    --bf16 \
    --train_adapter \
    --report_to="wandb" \
    $extra_args
  return 0
}

source /mnt/nas_home/dmf45/miniconda3/bin/activate base

languages=('en')
patterns=("xlm-roberta-base")
out_prefixes=("output_lora_peft/dynamic_tk")
methods=("lora")
dropouts=(0.3)
ranks=(128)
seeds=(42)
tokenizations=("dynamic")
merges=(20)
exp_types=("dynamic_bpe")

train "xlm-roberta-base" "output_lora_peft/dynamic_tk" "lora" 0.3 128 42 "dynamic" 20 "dynamic_bpe" "en"
train "xlm-roberta-base" "output_lora_peft/dynamic_tk" "lora" 0.3 32 42 "dynamic" 140 word_tk_hypernet "en"

for pattern in "${patterns[@]}"; do
  for out_prefix in "${out_prefixes[@]}"; do
    for method in "${methods[@]}"; do
      for dropout in "${dropouts[@]}"; do
        for rank in "${ranks[@]}"; do
          for seed in "${seeds[@]}"; do
            for exp_type in "${exp_types[@]}"; do
              for tokenization in "${tokenizations[@]}"; do
                for merge in "${merges[@]}"; do
                  for lng in "${languages[@]}"; do
                    train $pattern $out_prefix $method $dropout $rank $seed $tokenization $merge $exp_type $lng
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done
