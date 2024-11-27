#!/bin/bash
#SBATCH -J eval-uner
#SBATCH --time=20:0:0
#SBATCH --gres=gpu:1

JOBID=$SLURM_JOB_ID
EXP_NAME="dynamic_tokenization_UNER_eval"

source /mnt/nas_home/dmf45/miniconda3/bin/activate base

exp_types=("dynamic_bpe")

test_lngs=(
    "en_pud"
    "de_pud"
    "pt_bosque"
    "pt_pud"
    "ru_pud"
)

adapter_paths=("output_lora_peft/ner/dynamic_tk/UNER_75PERC__dynamic_bpe_en_lora_alpha_512_drop_0.3_rank_256_seed_42_merges_36_27062024_v2")
# adapter_paths=("output_lora_peft/ner/dynamic_tk/TOKENISER_SAMPLING_per_batch_UNIFORM_RETRY_2_dynamic_bpe_en_lora_alpha_512_drop_0.3_rank_256_seed_42_merges_154_02072024")

for adapter_path in "${adapter_paths[@]}"; do
    for lng in "${test_lngs[@]}"; do
        for exp_type in "${exp_types[@]}"; do
            python3 encoders/evaluation/uner/uner_evaluation.py --exp_type $exp_type --lng $lng --adapter_path $adapter_path --peft --multiple_merges_exp --batch_size 32 --bpe_token_boundary pretokens --best_adapter_criterion f1 --exp_prefix "UNER_word_level"
        done
    done
done
