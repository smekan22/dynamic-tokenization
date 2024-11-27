#!/bin/bash
#SBATCH -J eval-uner
#SBATCH --time=4:0:0
#SBATCH --gres=gpu:1

JOBID=$SLURM_JOB_ID
EXP_NAME="dynamic_tokenization_UNER_eval"

source /mnt/nas_home/dmf45/miniconda3/bin/activate base

exp_types=("plain" "fvt" "original_tk_hypernet" "word_tk_hypernet")

# plain - using original embeddings and tokenization
# fvt - using fvt embeddings and word-level tokenization
# original_tk_hypernet - using hypernetwork embeddings and original tokenization
# word_tk_hypernet - using hypernetwork embeddings and word-level tokenization

NAMES=(
    "en_ewt"
    "de_pud"
    "pt_bosque"
    "pt_pud"
    "ru_pud"
)

adapter_path="output_lora_peft/ner/UNER_subword_tk_en_lora_alpha_512_drop_0.3_rank_256_seed_42"

for lng in "${languages[@]}"; do
    for exp_type in "${exp_types[@]}"; do
        python3 uner_evaluation.py --exp_type $exp_type --lng $lng --adapter_path $adapter_path --peft
    done
done
