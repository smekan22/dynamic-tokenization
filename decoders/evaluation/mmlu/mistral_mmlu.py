from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import faiss
import pickle
from datasets import load_dataset
import datasets

import argparse
import sys
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import math
import random
import wandb
from collections import defaultdict
import numpy as np
import os
from evaluation_utils import get_hn_embeddings_for_tokens
from tokenizers.models import BPE, WordPiece

HOME_PATH = "/mnt/nas_home/dmf45/dynamic_tokenization"
sys.path.insert(0, HOME_PATH)

from tokenizations.static_tokenizations import tokenize
from tokenizations.tokenization_utils import DatasetEncoder
from tokenizations.hypernet_cache import LRU_Cache

parser = argparse.ArgumentParser(description="Running MMLU Evaluation")
parser.add_argument(
    "--ds_subject",
    type=str,
    default="all",
    help="The MMLU subject subset to use for evaluation.",
)
parser.add_argument(
    "--exp_type",
    type=str,
    default="plain",
    help="Choose which type of experiment to use: plain (original tokenization), original_tk_hypernet (HN embeddings), lp_tk_hypernet (longest prefix tokenization), dynamic_bpe (HN embeddings with different number of merges)",
)
parser.add_argument("--verbose", type=bool, default=False, help="Add extra loggings.")
parser.add_argument(
    "--eval_type",
    type=str,
    default="origianl",
    help="Original (compare with probs of A, B, C or D) or Harness (compare with probs of each choice text)",
)
parser.add_argument(
    "--batch_size", type=int, default=4, help="Batch size to use during evaluation"
)
parser.add_argument(
    "--five_shot", action="store_true", help="Perform 5-shots evaluation"
)
parser.add_argument("--no_wandb", action="store_true", help="Don't log data to wandb.")
parser.add_argument(
    "--exp_prefix",
    type=str,
    default="",
    help="Prefix to be added for the exp name in wandb",
)
parser.add_argument(
    "--same_domain_shot",
    action="store_true",
    help="Choose 5 shots from the same domain (but different split)",
)
parser.add_argument(
    "--lng", type=str, default="en", help="Language to use for hypernetwork"
)
parser.add_argument(
    "--max_len",
    type=int,
    default=4096,
    help="Max length to be used during tokenization",
)
parser.add_argument(
    "--vocab_1M",
    action="store_true",
    help="Use tokenizer with 1M vocab and HN embeddings",
)
parser.add_argument(
    "--multiple_merges_exp",
    action="store_true",
    help="Use HN embeddings with different perecentages of sequence reduction",
)
parser.add_argument(
    "--use_original_emb_for_choices",
    action="store_true",
    help="Use original embeddings for A, B, C, D choices",
)


args = parser.parse_args()
subject = args.ds_subject
exp_type = args.exp_type
verbose = args.verbose
eval_type = args.eval_type
batch_size = args.batch_size
five_shot = args.five_shot
exp_prefix = args.exp_prefix
same_domain_shot = args.same_domain_shot
lng = args.lng
max_length = args.max_len
use_vocab_1M = args.vocab_1M
use_original_emb_for_choices = args.use_original_emb_for_choices


def setup_seed(seed):
    random.seed(0)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


setup_seed(1234)

if not args.no_wandb:
    if exp_prefix != "":
        exp_prefix += "_"
    wandb.init(
        project="dynamic-tokenization",
        config={
            "dataset": "MMLU",
            "exp_type": exp_type,
            "eval_type": eval_type,
        },
        name=f"{exp_prefix}MMLU_Mistral_{exp_type}_{eval_type}_five_shot_{five_shot}_subject_{subject}_batch_size_{batch_size}_1M_Vocab_{use_vocab_1M}",
    )


class MMLUDataset(Dataset):
    def __init__(self, dataset, validation_dataset, validation_datasets, num_shots=5):
        self.dataset = dataset
        self.validation_dataset = validation_dataset
        self.num_shots = num_shots
        self.validation_datasets = validation_datasets

    def __len__(self):
        return len(self.dataset)

    def format_prompt(
        self,
        question,
        choices,
        subject: str = "",
        is_context_question: bool = False,
        same_domain_shot: bool = True,
        answer: str = "",
        five_shot: bool = False,
    ):
        subject = subject.replace("_", " ")
        if is_context_question:
            assert answer != ""
            if same_domain_shot:
                return f"This question refers to the following information.\n{question.strip()}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer: {answer}\n\n"
            else:  # random domain shots
                return f"This question is about {subject} and refers to the following information.\n{question.strip()}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer: {answer}\n\n"
        else:  # if main prompt question
            if five_shot and same_domain_shot:
                return f"This question refers to the following information.\n{question.strip()}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer:"
            elif five_shot and not same_domain_shot:
                return f"This question is about {subject} and refers to the following information.\n{question.strip()}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer:"
            return f"{question.strip()}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer:"

    def __getitem__(self, idx):
        item = self.dataset[idx]
        question = item["question"]
        choices = item["choices"]
        correct_answer_index = item["answer"]
        subject = item["subject"]
        context = ""
        if five_shot:
            for _ in range(self.num_shots):
                if not same_domain_shot:
                    example = random.choice(self.validation_dataset)
                else:
                    example = random.choice(self.validation_datasets[subject])
                while example["question"] == question and set(
                    example["choices"]
                ) == set(choices):
                    if not same_domain_shot:
                        example = random.choice(self.validation_dataset)
                    else:
                        example = random.choice(self.validation_datasets[subject])

                if example["question"] == question and set(example["choices"]) == set(
                    choices
                ):
                    raise Exception(
                        "Context question should be different than prompt question. Please check!"
                    )

                example_question = example["question"]
                example_choices = example["choices"]
                example_answer_index = example["answer"]
                example_answer = chr(65 + example_answer_index)
                if same_domain_shot:
                    assert example["subject"] == subject
                example_prompt = self.format_prompt(
                    question=example_question,
                    choices=example_choices,
                    is_context_question=True,
                    answer=example_answer,
                    same_domain_shot=same_domain_shot,
                    subject=example["subject"],
                )

                # context += f"{example_prompt} {example_answer}\n\n"
                context += example_prompt

        prompt = context + self.format_prompt(
            question=question,
            choices=choices,
            subject=subject,
            five_shot=five_shot,
            same_domain_shot=same_domain_shot,
        )
        # If shots are from same domain or if we are only doing 0 shot evaluation
        if (five_shot and same_domain_shot) or (not five_shot):
            subject = subject = subject.replace("_", " ")
            prompt = f"The following are multiple choice questions (with answers) about {subject}.\n\n{prompt}"
        elif (
            five_shot and not same_domain_shot
        ):  # Random shots - not necessarily same domain
            prompt = f"The following are multiple choice questions (with answers).\n\n{prompt}"
        init_prompt = self.format_prompt(
            question=question,
            choices=choices,
            subject=subject,
            five_shot=five_shot,
            same_domain_shot=same_domain_shot,
        )
        return prompt, choices, correct_answer_index, context, init_prompt, subject


def collate_fn(batch):
    prompts = [item[0] for item in batch]
    choices = [item[1] for item in batch]
    correct_answer_indices = [item[2] for item in batch]
    contexts = [item[3] for item in batch]
    init_prompts = [item[4] for item in batch]
    subjects = [item[5] for item in batch]
    return prompts, choices, correct_answer_indices, contexts, init_prompts, subjects


def evaluate_model(dataloader):
    correct_predictions = 0
    total_questions = 0
    seq_lens = []

    if eval_type == "original":
        choices_softmax = ["A", "B", "C", "D"]
        choices_softmax = [tokenizer.tokenize(choice)[0] for choice in choices_softmax]
        if exp_type == "plain":
            print("Using original embeddings for A, B, C, D choices")
            choices_softmax_ids = torch.tensor(
                tokenizer.convert_tokens_to_ids(choices_softmax)
            )
        elif (
            exp_type == "original_tk_hypernet"
            or exp_type == "lp_tk_hypernet"
            or exp_type == "dynamic_bpe"
        ):
            if use_vocab_1M:
                tokenizer_to_use = AutoTokenizer.from_pretrained(
                    "benjamin/zett-hypernetwork-Mistral-7B-v0.1"
                )
            else:
                tokenizer_to_use = tokenizer
            if use_original_emb_for_choices:
                print("Using original embeddings!")
                choices_softmax_ids = torch.tensor(
                    tokenizer.convert_tokens_to_ids(choices_softmax)
                )
                answer_output_embeddings = (
                    base_model.get_output_embeddings()
                    .weight.data[choices_softmax_ids]
                    .to(device)
                    .to(torch.bfloat16)
                )
            else:

                _, answer_output_embeddings = get_hn_embeddings_for_tokens(
                    tokens=choices_softmax,
                    tokenizer=tokenizer_to_use,
                    lang_index=lang_index,
                    hypernet=hypernet,
                    source_embeddings=source_embeddings,
                    device=device,
                    base_input_embeddings=base_model.get_input_embeddings().weight.data.to(
                        device
                    ),
                    base_output_embeddings=base_model.get_output_embeddings().weight.data.to(
                        device
                    ),
                )

    if (
        exp_type == "dynamic_bpe"
    ):  # Evaluate MMLU - using HN embeddings and dynamic tokenization for prefilling
        seqReduction2Accuracy = {}
        seqReduction2seqLen = {}
        merges2seqLen = {}
        merges2accuracy = {}
        merges2seqRed = {
            0: 0,  # 0 merges for 0% reduction
            1: 10,  # 1 merge for 10% reduction
            3: 20,  # 3 merges for 20% reduction
            7: 30,  # 7 merges for 30% reduction
            11: 40,  # 11 merges for 40% reduction
            18: 50,  # 18 merges for 50% reduction
            26: 60,  # 26 merges for 60% reduction
            38: 70,  # 38 merges for 70% reduction
            58: 80,  # 58 merges for 80% reduction
            98: 90,  # 98 merges for 90% reduction
            250: 100,  # 250 merges for 100% reduction
        }

        avg_length = 0

        for merges in merges2seqRed:
            print(f"Encoding end evaluating with {merges} merges...")
            embeddings_cache = LRU_Cache(
                cache_size=5000,
                emb_size=base_model.get_input_embeddings().weight.data.shape[1],
                device=device,
            )
            datasetEncoder.reset_state(embeddings_cache=embeddings_cache)

            correct_predictions = 0
            total_questions = 0
            seq_lens = []

            for batch in tqdm(dataloader, desc="Evaluating"):
                (
                    prompts,
                    choices,
                    correct_answer_indices,
                    contexts,
                    init_prompts,
                    subjects,
                ) = batch
                inputs = datasetEncoder.encode_examples_unique_tokens_lru(
                    examples=prompts, task="mmlu", max_length=max_length, merges=merges
                )

                inputs.pop("batch_tokens", None)
                inputs["inputs_embeds"] = inputs["inputs_embeds"].to(torch.bfloat16)
                for ids_sample in inputs["inputs_embeds"]:
                    seq_lens.append(len(ids_sample))
                if (
                    "input_ids" in inputs and inputs["input_ids"].shape[1] > max_length
                ) or (
                    "inputs_embeds" in inputs
                    and inputs["inputs_embeds"].shape[1] > max_length
                ):
                    raise Exception(
                        "Output is truncated, hence new question is truncated. It needs to be fixed."
                    )

                with torch.no_grad():
                    outputs = model(**inputs, output_hidden_states=True)
                    logits = outputs.logits

                last_token_logits = logits[:, -1, :]
                probabilities = torch.softmax(last_token_logits, dim=-1)

                for i in range(len(prompts)):
                    if eval_type == "original":
                        last_hidden_state = outputs.hidden_states[-1][:, -1, :][i]
                        logits = last_hidden_state @ answer_output_embeddings.T
                        answer_probs = torch.softmax(logits, dim=-1)
                        predicted_answer_index = torch.argmax(answer_probs).item()
                    elif eval_type == "harness":
                        raise NotImplementedError(
                            "Harness not yet implemented for dynamic bpe experiments."
                        )

                    correct_answer_index = correct_answer_indices[i]
                    if predicted_answer_index == correct_answer_index:
                        correct_predictions += 1

                    total_questions += 1

                    if verbose:
                        print(f"Prompt: {prompts[i]}")
                        print(f"Correct Answer Index: {correct_answer_index}")
                        print(f"Predicted Answer Index: {predicted_answer_index}")
                        print(f"Answer Probs: {answer_probs}")
                        print()

            seq_lengths = datasetEncoder.seq_lengths
            curr_avg_length = sum(seq_lengths) / len(seq_lengths)
            print("Current length", curr_avg_length)

            merges2seqLen[merges] = curr_avg_length
            seqReduction2seqLen[merges2seqLen[merges]] = curr_avg_length
            avg_length = curr_avg_length
            accuracy = correct_predictions / total_questions * 100
            merges2accuracy[merges] = accuracy
            seqReduction2Accuracy[merges2seqRed[merges]] = accuracy
            print(f"Merges {merges} | Seq reduction {merges2seqRed[merges]}")
            print(f"Avg. sequence lengths {avg_length}")
            print(f"Accuracy {accuracy}")

        print("Merges2Accuracy", merges2accuracy)
        print("seqReduction2Accuracy", seqReduction2Accuracy)
        print("seqReduction2seqLen", seqReduction2seqLen)

        data_merges2seqLengths = [
            [nr_merges, length] for nr_merges, length in merges2seqLen.items()
        ]
        table_merges2seqLengths = wandb.Table(
            data=data_merges2seqLengths,
            columns=["Number of Merges", "Avg Sequence Length"],
        )

        data_merges2accuracy = [
            [nr_merges, accuracy] for nr_merges, accuracy in merges2accuracy.items()
        ]
        table_merges2accuracy = wandb.Table(
            data=data_merges2accuracy, columns=["Number of Merges", "Accuracy"]
        )

        data_seqReduction2accuracy = [
            [seq_red, accuracy] for seq_red, accuracy in seqReduction2Accuracy.items()
        ]
        table_seqReduction2accuracy = wandb.Table(
            data=data_seqReduction2accuracy, columns=["Seq Reduction", "Accuracy"]
        )

        data_accuracy_vs_seqLength = [
            [merges2seqLen[merges], merges2accuracy[merges]]
            for merges in merges2accuracy.keys()
        ]
        table_accuracy_vs_seqLength = wandb.Table(
            data=data_accuracy_vs_seqLength, columns=["Avg Sequence Length", "Accuracy"]
        )

        if not args.no_wandb:
            wandb.log(
                {
                    "Accuracy vs. Number of Merges": wandb.plot.line(
                        table_merges2accuracy,
                        "Number of Merges",
                        "Accuracy",
                        title="Accuracy vs. Number of Merges",
                    ),
                    "Average Sequence Length vs. Number of Merges": wandb.plot.line(
                        table_merges2seqLengths,
                        "Number of Merges",
                        "Avg Sequence Length",
                        title="Average Sequence Length vs. Number of Merges",
                    ),
                    "Accuracy vs. Avg Sequence Length": wandb.plot.line(
                        table_accuracy_vs_seqLength,
                        "Avg Sequence Length",
                        "Accuracy",
                        title="Accuracy vs. Avg Sequence Length",
                    ),
                    "Accuracy vs. Seq Reduction": wandb.plot.line(
                        table_seqReduction2accuracy,
                        "Seq Reduction",
                        "Accuracy",
                        title="Accuracy vs. Seq Reduction",
                    ),
                }
            )
            wandb.finish()
    else:  # Evaluate MMLU using one of [ original tokenization with original embeddings, original tokenization with HN embeddings, word tokenization with HN embeddings, LP tokenization with HN embeddings ]
        results_table = wandb.Table(
            columns=[
                "Question",
                "Predicted Answer",
                "Correct Answer",
                "Is correct?",
                "Context",
            ]
        )

        subject2correct = defaultdict(int)
        subject2totalQuestions = defaultdict(int)

        for batch in tqdm(dataloader, desc="Evaluating"):
            (
                prompts,
                choices,
                correct_answer_indices,
                contexts,
                init_prompts,
                subjects,
            ) = batch

            if exp_type == "plain":
                inputs = tokenizer(
                    prompts,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                    padding=True,
                ).to(device)
                for ids_sample in inputs["input_ids"]:
                    seq_lens.append(len(ids_sample))
            elif exp_type == "lp_tk_hypernet" and use_vocab_1M:
                with open(
                    "decoders/data/1M_vocab_embeddings/token2id.pkl", "rb"
                ) as pickle_file:
                    token2id = pickle.load(pickle_file)
                tokenizer.eos_token_id = token2id["</s>"]
                tokenizer.bos_token_id = token2id["<s>"]
                tokenizer.pad_token_id = token2id["</s>"]
                tokenizer.unk_token_id = token2id["<unk>"]
                tokenizer.eos_token = "</s>"
                tokenizer.bos_token = "<s>"
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.unk_token = "<unk>"

                inputs_tokens = [
                    ["<s>"]
                    + tokenize(
                        prompt, tokenizer, max_length=max_length, truncation=True
                    )
                    for prompt in prompts
                ]
                input_ids = [
                    [token2id[token] for token in current_sample_in_batch]
                    for current_sample_in_batch in inputs_tokens
                ]

                max_batch_length = -1
                for ids_sample in input_ids:
                    max_batch_length = max(max_batch_length, len(ids_sample))

                # attention masks and batch embeddings
                attention_masks = []
                batch_embeddings = []
                for ids_sample in input_ids:
                    seq_lens.append(len(ids_sample))
                    attention_ones = len(ids_sample)
                    pad_tokens_nr = max_batch_length - attention_ones
                    ids_sequence = [token2id["</s>"]] * (pad_tokens_nr) + ids_sample
                    attention_zeros = pad_tokens_nr
                    attention_mask = torch.tensor(
                        [0] * attention_zeros + [1] * attention_ones, device=device
                    )
                    attention_masks.append(attention_mask)
                    batch_embeddings.append(
                        inout_1M_embeddings["input_embeddings"][ids_sequence]
                        .to(device)
                        .to(torch.bfloat16)
                    )

                inputs = {}
                inputs["attention_mask"] = torch.stack(attention_masks).to(device)
                inputs["inputs_embeds"] = torch.stack(batch_embeddings).to(device)
            elif (
                exp_type == "original_tk_hypernet"
                or exp_type == "lp_tk_hypernet"
                or exp_type == "word_tk_hypernet"
            ):
                inputs = datasetEncoder.encode_examples_unique_tokens_lru(
                    examples=prompts, task="mmlu", max_length=max_length
                )
                inputs["inputs_embeds"] = inputs["inputs_embeds"].to(torch.bfloat16)
                for ids_sample in inputs["inputs_embeds"]:
                    seq_lens.append(len(ids_sample))
            if (
                "input_ids" in inputs and inputs["input_ids"].shape[1] > max_length
            ) or (
                "inputs_embeds" in inputs
                and inputs["inputs_embeds"].shape[1] > max_length
            ):
                raise Exception(
                    "Output is truncated, hence new question is truncated. It needs to be fixed."
                )

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                logits = outputs.logits

            last_token_logits = logits[:, -1, :]
            probabilities = torch.softmax(last_token_logits, dim=-1)

            for i in range(len(prompts)):
                if eval_type == "original":
                    if exp_type == "plain":
                        answer_probs = probabilities[i, choices_softmax_ids]
                        predicted_answer_index = torch.argmax(answer_probs).item()
                    elif (
                        exp_type == "original_tk_hypernet"
                        or exp_type == "lp_tk_hypernet"
                    ):
                        last_hidden_state = outputs.hidden_states[-1][:, -1, :][i]
                        logits = last_hidden_state @ answer_output_embeddings.T
                        answer_probs = torch.softmax(logits, dim=-1)
                        predicted_answer_index = torch.argmax(answer_probs).item()
                elif eval_type == "harness":
                    answer_probs = []
                    for j, choice in enumerate(choices[i]):
                        formatted_choice = f"{chr(65+j)}. {choice}"
                        answer_inputs_ids = tokenizer(
                            formatted_choice, return_tensors="pt"
                        )["input_ids"].to(device)
                        answer_token_probability = probabilities[i, answer_inputs_ids]
                        answer_token_probability_log = torch.log(
                            answer_token_probability
                        )
                        answer_log_probability = (
                            answer_token_probability_log.sum().item()
                            / answer_token_probability.shape[1]
                        )
                        answer_probs.append(answer_log_probability)

                    predicted_answer_index = torch.argmax(
                        torch.tensor(answer_probs)
                    ).item()

                    if predicted_answer_index > 3:
                        raise Exception("This is impossible for MMLU!")

                correct_answer_index = correct_answer_indices[i]
                is_correct = "NO"
                if predicted_answer_index == correct_answer_index:
                    correct_predictions += 1
                    is_correct = "YES"
                    subject2correct[subjects[i]] += 1

                subject2totalQuestions[subjects[i]] += 1
                results_table.add_data(
                    init_prompts[i],
                    predicted_answer_index,
                    correct_answer_index,
                    is_correct,
                    contexts[i],
                )

                total_questions += 1

                if verbose:
                    print(f"Prompt: {prompts[i]}")
                    print(f"Correct Answer Index: {correct_answer_index}")
                    print(f"Predicted Answer Index: {predicted_answer_index}")
                    print(f"Answer Probs: {answer_probs}")
                    print()

        accuracy = correct_predictions / total_questions * 100

        subject_accuracies = {
            subject: subject2correct[subject] / subject2totalQuestions[subject] * 100
            for subject in subject2totalQuestions
        }

        if not args.no_wandb:
            for subject in subject_accuracies:
                wandb.log({f"{subject}_accuracy": subject_accuracies[subject]})

            subject_accuracy_data = [
                [subject, acc] for subject, acc in subject_accuracies.items()
            ]
            subject_accuracy_table = wandb.Table(
                data=subject_accuracy_data, columns=["Subject", "Accuracy"]
            )
            subject_accuracy_plot = wandb.plot.bar(
                subject_accuracy_table,
                "Subject",
                "Accuracy",
                title="Per-Subject Accuracies",
            )

            wandb.log({"Subject Accuracies Plot": subject_accuracy_plot})

            if len(seq_lens):
                avg_seq_len = sum(seq_lens) / len(seq_lens)
            else:
                avg_seq_len = 0
            print(f"Avg seq len: {avg_seq_len}")
            wandb.log(
                {
                    f"ALL_results_table": results_table,
                    "ALL_Accuracy": accuracy,
                    "Avg_seq_len": avg_seq_len,
                }
            )
            wandb.finish()
        return avg_seq_len


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set pad token
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(
    device
)

if exp_type in [
    "original_tk_hypernet",
    "lp_tk_hypernet",
    "dynamic_bpe",
    "word_tk_hypere",
]:
    hypernet = AutoModel.from_pretrained(
        "benjamin/zett-hypernetwork-Mistral-7B-v0.1", trust_remote_code=True
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        "benjamin/zett-hypernetwork-Mistral-7B-v0.1"
    )
    if exp_type == "lp_tk_hypernet":
        if use_vocab_1M:
            inout_1M_embeddings = torch.load(
                "decoders/data/1M_vocab_embeddings/large_HN_embeddings.pt"
            )
            vocab_new_path = "decoders/data/large_tokenizer/vocab.json"
            merges_path = "decoders/data/large_tokenizer/merges.txt"
            vocab, merges = BPE.read_file(vocab_new_path, merges_path)
        else:
            vocab = tokenizer.get_vocab()
        unk_token = (
            tokenizer.unk_token
            if tokenizer.unk_token is not None
            else tokenizer.eos_token
        )
        # use WordPiece without prefix to achieve longest-prefix tokenization
        tokenizer._tokenizer.model = WordPiece(vocab, unk_token=unk_token)
        tokenizer._tokenizer.model.continuing_subword_prefix = ""

    langs = [x.strip() for x in open("artifacts/26l.txt")]
    lang_index = torch.tensor(langs.index(lng), dtype=torch.int32).to(device)

    base_model = AutoModelForCausalLM.from_pretrained(model_name)

    source_embeddings = torch.concatenate(
        [
            base_model.get_input_embeddings().weight.data,
            base_model.get_output_embeddings().weight.data,
        ],
        axis=1,
    ).to(device)
    embeddings_cache = LRU_Cache(
        cache_size=5000,
        emb_size=base_model.get_input_embeddings().weight.data.shape[1],
        device=device,
    )
    datasetEncoder = DatasetEncoder(
        hypernet=hypernet,
        tokenizer=tokenizer,
        device=device,
        lang_index=lang_index,
        surface_form_maxlen=7,
        source_embeddings=source_embeddings,
        embeddings_cache=embeddings_cache,
        exp_type=exp_type,
        collect_extra_data=True,
        bpe_tokenizer_boundary="pretokens",
    )

subjects = [
    "abstract_algebra",
    "high_school_government_and_politics",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]

dataset = load_dataset("hails/mmlu_no_train", subject)["test"]
validation_dataset = load_dataset("hails/mmlu_no_train", subject)["validation"]

per_subject_validation_datasets = {}
if same_domain_shot:
    max_length = max(8192, max_length)
    for subject in subjects:
        try:
            print(f"Downloading {subject} data")
            per_subject_validation_datasets[subject] = load_dataset(
                "cais/mmlu", subject, cache_dir="~/.cache/huggingface/datasets"
            )["validation"]
        except:
            raise Exception(f"Error when downloading dataset for subject {subject}")

mmlu_dataset = MMLUDataset(
    dataset,
    validation_dataset=validation_dataset,
    validation_datasets=per_subject_validation_datasets,
)
dataloader = DataLoader(
    mmlu_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
)

accuracy = evaluate_model(dataloader)
print(accuracy)
