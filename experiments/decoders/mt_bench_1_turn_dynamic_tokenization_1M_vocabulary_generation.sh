#!/bin/bash
#SBATCH -J mt_bench_mistral
#SBATCH --time=30:0:0
#SBATCH --gres=gpu:1
#SBATCH --nodelist=ltl-gpu05

JOBID=$SLURM_JOB_ID
EXP_NAME="mt_bench_mistral"

source /mnt/nas_home/dmf45/miniconda3/bin/activate base

python3 FastChat/fastchat/llm_judge/gen_model_answer.py --model-path mistralai/Mistral-7B-Instruct-v0.1 --model-id 1_TURN_NOV_2024_1M_vocab_gen_32k_vocab_0_merges_0perc_rep11_eos1 --dynamic_bpe_merges 0 --use_hn_emb --repetition_penalty 1.1 --min_p 0.1 --use_top_k 
python3 FastChat/fastchat/llm_judge/gen_judgment.py --model-list 1_TURN_NOV_2024_1M_vocab_gen_32k_vocab_0_merges_0perc_rep11_eos1 --judge-model gpt-3.5-turbo-1106
python3 FastChat/fastchat/llm_judge/show_result.py --judge-model gpt-3.5-turbo-1106 --model-list 1_TURN_NOV_2024_1M_vocab_gen_32k_vocab_0_merges_0perc_rep11_eos1

python3 FastChat/fastchat/llm_judge/gen_model_answer.py --model-path mistralai/Mistral-7B-Instruct-v0.1 --model-id 1_TURN_NOV_2024_1M_vocab_gen_32k_vocab_2_merges_10perc --dynamic_bpe_merges 2 --use_hn_emb --repetition_penalty 1.1  --min_p 0.1 --use_top_k 
python3 FastChat/fastchat/llm_judge/gen_judgment.py --model-list 1_TURN_NOV_2024_1M_vocab_gen_32k_vocab_2_merges_10perc --judge-model gpt-3.5-turbo-1106
python3 FastChat/fastchat/llm_judge/show_result.py --judge-model gpt-3.5-turbo-1106 --model-list 1_TURN_NOV_2024_1M_vocab_gen_32k_vocab_2_merges_10perc

python3 FastChat/fastchat/llm_judge/gen_model_answer.py --model-path mistralai/Mistral-7B-Instruct-v0.1 --model-id 1_TURN_NOV_2024_1M_vocab_gen_32k_vocab_6_merges_20perc --dynamic_bpe_merges 6 --use_hn_emb --repetition_penalty 1.1  --min_p 0.1 --use_top_k 
python3 FastChat/fastchat/llm_judge/gen_judgment.py --model-list 1_TURN_NOV_2024_1M_vocab_gen_32k_vocab_6_merges_20perc --judge-model gpt-3.5-turbo-1106
python3 FastChat/fastchat/llm_judge/show_result.py --judge-model gpt-3.5-turbo-1106 --model-list 1_TURN_NOV_2024_1M_vocab_gen_32k_vocab_6_merges_20perc

python3 FastChat/fastchat/llm_judge/gen_model_answer.py --model-path mistralai/Mistral-7B-Instruct-v0.1 --model-id 1_TURN_NOV_2024_1M_vocab_gen_32k_vocab_11_merges_30perc --dynamic_bpe_merges 11 --use_hn_emb --repetition_penalty 1.1  --min_p 0.1 --use_top_k 
python3 FastChat/fastchat/llm_judge/gen_judgment.py --model-list 1_TURN_NOV_2024_1M_vocab_gen_32k_vocab_11_merges_30perc --judge-model gpt-3.5-turbo-1106
python3 FastChat/fastchat/llm_judge/show_result.py --judge-model gpt-3.5-turbo-1106 --model-list 1_TURN_NOV_2024_1M_vocab_gen_32k_vocab_11_merges_30perc

python3 FastChat/fastchat/llm_judge/gen_model_answer.py --model-path mistralai/Mistral-7B-Instruct-v0.1 --model-id 1_TURN_NOV_2024_1M_vocab_gen_32k_vocab_20_merges_40perc --dynamic_bpe_merges 20 --use_hn_emb --repetition_penalty 1.1  --min_p 0.1 --use_top_k 
python3 FastChat/fastchat/llm_judge/gen_judgment.py --model-list 1_TURN_NOV_2024_1M_vocab_gen_32k_vocab_20_merges_40perc --judge-model gpt-3.5-turbo-1106
python3 FastChat/fastchat/llm_judge/show_result.py --judge-model gpt-3.5-turbo-1106 --model-list 1_TURN_NOV_2024_1M_vocab_gen_32k_vocab_20_merges_40perc

python3 FastChat/fastchat/llm_judge/gen_model_answer.py --model-path mistralai/Mistral-7B-Instruct-v0.1 --model-id 1_TURN_NOV_2024_1M_vocab_gen_32k_vocab_29_merges_50perc --dynamic_bpe_merges 29 --use_hn_emb --repetition_penalty 1.1  --min_p 0.1 --use_top_k 
python3 FastChat/fastchat/llm_judge/gen_judgment.py --model-list 1_TURN_NOV_2024_1M_vocab_gen_32k_vocab_29_merges_50perc --judge-model gpt-3.5-turbo-1106
python3 FastChat/fastchat/llm_judge/show_result.py --judge-model gpt-3.5-turbo-1106 --model-list 1_TURN_NOV_2024_1M_vocab_gen_32k_vocab_29_merges_50perc

python3 FastChat/fastchat/llm_judge/gen_model_answer.py --model-path mistralai/Mistral-7B-Instruct-v0.1 --model-id 1_TURN_NOV_2024_1M_vocab_gen_32k_vocab_39_merges_60perc --dynamic_bpe_merges 39 --use_hn_emb --repetition_penalty 1.1  --min_p 0.1 --use_top_k 
python3 FastChat/fastchat/llm_judge/gen_judgment.py --model-list 1_TURN_NOV_2024_1M_vocab_gen_32k_vocab_39_merges_60perc --judge-model gpt-3.5-turbo-1106
python3 FastChat/fastchat/llm_judge/show_result.py --judge-model gpt-3.5-turbo-1106 --model-list 1_TURN_NOV_2024_1M_vocab_gen_32k_vocab_39_merges_60perc

python3 FastChat/fastchat/llm_judge/gen_model_answer.py --model-path mistralai/Mistral-7B-Instruct-v0.1 --model-id 1_TURN_NOV_2024_1M_vocab_gen_32k_vocab_48_merges_70perc --dynamic_bpe_merges 48 --use_hn_emb --repetition_penalty 1.1  --min_p 0.1 --use_top_k 
python3 FastChat/fastchat/llm_judge/gen_judgment.py --model-list 1_TURN_NOV_2024_1M_vocab_gen_32k_vocab_48_merges_70perc --judge-model gpt-3.5-turbo-1106
python3 FastChat/fastchat/llm_judge/show_result.py --judge-model gpt-3.5-turbo-1106 --model-list 1_TURN_NOV_2024_1M_vocab_gen_32k_vocab_48_merges_70perc

python3 FastChat/fastchat/llm_judge/gen_model_answer.py --model-path mistralai/Mistral-7B-Instruct-v0.1 --model-id 1_TURN_NOV_2024_1M_vocab_gen_32k_vocab_57_merges_80perc --dynamic_bpe_merges 57 --use_hn_emb --repetition_penalty 1.1  --min_p 0.1 --use_top_k 
python3 FastChat/fastchat/llm_judge/gen_judgment.py --model-list 1_TURN_NOV_2024_1M_vocab_gen_32k_vocab_57_merges_80perc --judge-model gpt-3.5-turbo-1106
python3 FastChat/fastchat/llm_judge/show_result.py --judge-model gpt-3.5-turbo-1106 --model-list 1_TURN_NOV_2024_1M_vocab_gen_32k_vocab_57_merges_80perc

python3 FastChat/fastchat/llm_judge/gen_model_answer.py --model-path mistralai/Mistral-7B-Instruct-v0.1 --model-id 1_TURN_NOV_2024_1M_vocab_gen_32k_vocab_66_merges_90perc --dynamic_bpe_merges 66 --use_hn_emb --repetition_penalty 1.1  --min_p 0.1 --use_top_k 
python3 FastChat/fastchat/llm_judge/gen_judgment.py --model-list 1_TURN_NOV_2024_1M_vocab_gen_32k_vocab_66_merges_90perc --judge-model gpt-3.5-turbo-1106
python3 FastChat/fastchat/llm_judge/show_result.py --judge-model gpt-3.5-turbo-1106 --model-list 1_TURN_NOV_2024_1M_vocab_gen_32k_vocab_66_merges_90perc

python3 FastChat/fastchat/llm_judge/gen_model_answer.py --model-path mistralai/Mistral-7B-Instruct-v0.1 --model-id 1_TURN_NOV_2024_1M_vocab_gen_32k_vocab_145_merges_100perc --dynamic_bpe_merges 145 --use_hn_emb --repetition_penalty 1.1  --min_p 0.1 --use_top_k 
python3 FastChat/fastchat/llm_judge/gen_judgment.py --model-list 1_TURN_NOV_2024_1M_vocab_gen_32k_vocab_145_merges_100perc --judge-model gpt-3.5-turbo-1106
python3 FastChat/fastchat/llm_judge/show_result.py --judge-model gpt-3.5-turbo-1106 --model-list 1_TURN_NOV_2024_1M_vocab_gen_32k_vocab_145_merges_100perc