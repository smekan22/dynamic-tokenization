#!/bin/bash
#SBATCH -J uner_adapter_sampled_tokenisers
#SBATCH --time=40:0:0
#SBATCH --gres=gpu:1


train_uniform() {
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

  python scripts/adapters/uner/run_ner_dynamic.py \
    --model_name_or_path $(printf $pattern "en") \
    --push_to_hub False \
    --dataset_name="universalner/universal_ner" \
    --dataset_config_name="en_ewt" \
    --do_train \
    --do_eval \
    --do_tokeniser_sampling_per_batch \
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
    --output_dir $out_prefix/UNIFORM_TOKENISER_SAMPLING_per_batch_${prefix}_${exp_type}_${lng}_${method}_alpha_${alpha}_drop_${dropout}_rank_${rank}_seed_${seed}_merges_${dynamic_tokenization_merges}_${today} \
    --overwrite_output_dir \
    --bf16 \
    --train_adapter \
    --report_to="wandb" \
    $extra_args
  

    source /mnt/nas_home/dmf45/miniconda3/bin/activate base
    exp_types=("dynamic_bpe")
    adapter_paths=($out_prefix/UNIFORM_TOKENISER_SAMPLING_per_batch_${prefix}_${exp_type}_${lng}_${method}_alpha_${alpha}_drop_${dropout}_rank_${rank}_seed_${seed}_merges_${dynamic_tokenization_merges}_${today})

    test_lngs=(
        "en_ewt"
    )

    for adapter_path in "${adapter_paths[@]}"; do
        for lng in "${test_lngs[@]}"; do
            for exp_type in "${exp_types[@]}"; do
                python3 uner_evaluation.py --exp_type $exp_type --lng $lng --adapter_path $adapter_path --peft --multiple_merges_exp --batch_size 32 --bpe_token_boundary pretokens --best_adapter_criterion f1 --exp_prefix "UNIFORM_${prefix}"
            done
        done
    done

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
    --do_tokeniser_sampling_per_batch_gaussian \
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
    --output_dir $out_prefix/GAUSSIAN_TOKENISER_SAMPLING_per_batch_${prefix}_${exp_type}_${lng}_${method}_alpha_${alpha}_drop_${dropout}_rank_${rank}_seed_${seed}_merges_${dynamic_tokenization_merges}_${today} \
    --overwrite_output_dir \
    --bf16 \
    --train_adapter \
    --report_to="wandb" \
    $extra_args
  


    source /mnt/nas_home/dmf45/miniconda3/bin/activate base
    exp_types=("dynamic_bpe")
    adapter_paths=($out_prefix/GAUSSIAN_TOKENISER_SAMPLING_per_batch_${prefix}_${exp_type}_${lng}_${method}_alpha_${alpha}_drop_${dropout}_rank_${rank}_seed_${seed}_merges_${dynamic_tokenization_merges}_${today})

    test_lngs=(
        "en_ewt"
    )

    for adapter_path in "${adapter_paths[@]}"; do
        for lng in "${test_lngs[@]}"; do
            for exp_type in "${exp_types[@]}"; do
                python3 uner_evaluation.py --exp_type $exp_type --lng $lng --adapter_path $adapter_path --peft --multiple_merges_exp --batch_size 32 --bpe_token_boundary pretokens --best_adapter_criterion f1 --exp_prefix "GAUSSIAN_${prefix}"
            done
        done
    done
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
merges=(20 40 10 0)
exp_types=("dynamic_bpe")

train_uniform "xlm-roberta-base" "output_lora_peft/ner/dynamic_tk" "lora" 0.3 256 42 "dynamic" 154 "dynamic_bpe" "en" "UNIFORM"
train_gaussian "xlm-roberta-base" "output_lora_peft/ner/dynamic_tk" "lora" 0.3 256 42 "dynamic" 154 "dynamic_bpe" "en" "GAUSSIAN"

