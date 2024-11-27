#!/bin/bash
#SBATCH -J dynamic-t-eval-xnli
#SBATCH --time=5:0:0
#SBATCH --gres=gpu:1

JOBID=$SLURM_JOB_ID
EXP_NAME="dynamic_tokenization_xnli_EN_sentence_bpe_tokenization"

source /mnt/nas_home/dmf45/miniconda3/bin/activate base

exp_types=("dynamic_bpe")

languages=('en' 'ar' 'bg' 'de' 'el' 'es' 'fr' 'hi' 'ru' 'sw' 'tr' 'ur' 'vi')

# Uniform distributon adapter
adapter_path="output_lora_peft/dynamic_tk/TOKENISER_SAMPLING_per_batch_dynamic_bpe_en_lora_alpha_256_drop_0.3_rank_128_seed_42_merges_0_22062024/checkpoint-200000"

for lng in "${languages[@]}"; do
    for exp_type in "${exp_types[@]}"; do
        python3 encoders/evaluation/xnli/xnli_evaluation.py --exp_type $exp_type --lng $lng --multiple_merges_exp --batch_size 32 --bpe_token_boundary pretokens --adapter_path $adapter_path --peft --exp_prefix "GAUSSIAN_SAMPLING_per_batch_185k_R128" #> logs/xnli_$today_date/out.xnli_${today_date}.${lng}.${exp_type}.${JOBID}
    done
done
