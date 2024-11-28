from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from transformers import AutoTokenizer
from datasets import load_dataset
import os
import json


def load_merges(file_path):
    merges = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 2:
                merges.append((parts[0], parts[1]))
    return merges


def load_vocab(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        vocab = json.load(file)
    return vocab


hn_tokenizer = AutoTokenizer.from_pretrained(
    "benjamin/zett-hypernetwork-Mistral-7B-v0.1"
)
tokenizer = hn_tokenizer.backend_tokenizer
bpe_model = tokenizer.model

# Paths to Mistral 7B initial vocabulary
old_vocab_path = "decoders/data/tokenizer_hn_mistral/vocab.json"
old_merges_path = "decoders/data/tokenizer_hn_mistral/merges.txt"

vocab = load_vocab(old_vocab_path)
merges = load_merges(old_merges_path)


vocab, merges = BPE.read_file(old_vocab_path, old_merges_path)
new_bpe_model = BPE(vocab=vocab, merges=merges)
new_tokenizer = Tokenizer(new_bpe_model)


# Setup dataset
lng = "en"
raw_datasets = {}
raw_datasets["train"] = load_dataset(
    "parquet",
    data_files=os.path.join(
        "/mnt/nas_home/dmf45/disks/persist", "train", f"{lng}.parquet"
    ),
    split="train",
)
raw_datasets["train"] = raw_datasets["train"]
additional_texts = raw_datasets["train"]["text"]

# Configure the trainer to expand the vocabulary to 1M
trainer = BpeTrainer(
    vocab_size=1_000_000,
    show_progress=True,
    special_tokens=list(hn_tokenizer.special_tokens_map.values())
    + list(hn_tokenizer.vocab.keys()),
)

new_tokenizer.pre_tokenizer = hn_tokenizer.backend_tokenizer.pre_tokenizer
new_tokenizer.normalizer = hn_tokenizer.backend_tokenizer.normalizer

new_tokenizer.train_from_iterator(
    additional_texts, trainer=trainer, length=len(additional_texts)
)
new_tokenizer.model.save("decoders/data/large_tokenizer")

new_vocab_size = new_tokenizer.get_vocab_size()
print(f"New vocabulary size: {new_vocab_size}")
