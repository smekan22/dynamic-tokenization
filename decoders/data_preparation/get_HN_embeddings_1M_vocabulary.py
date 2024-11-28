import nmslib
import numpy as np
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from zett.utils import get_surface_form_matrix
import json
import torch
from tqdm import tqdm
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and hypernetwork
base_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
hypernet = AutoModel.from_pretrained(
    "benjamin/zett-hypernetwork-Mistral-7B-v0.1", trust_remote_code=True
).to(device)
hn_tokenizer = AutoTokenizer.from_pretrained(
    "benjamin/zett-hypernetwork-Mistral-7B-v0.1"
)
source_embeddings = torch.concatenate(
    [
        base_model.get_input_embeddings().weight.data,
        base_model.get_output_embeddings().weight.data,
    ],
    axis=1,
)
source_embeddings = source_embeddings.to(device)

langs = [x.strip() for x in open("artifacts/26l.txt")]
lang_index = torch.tensor(langs.index("en"), dtype=torch.int32).to(device)

vocab_init_path = "decoders/data/tokenizer_hn_mistral/vocab.json"
vocab_new_path = "decoders/data/large_tokenizer/vocab.json"

with open(vocab_init_path, "r") as file:
    vocab_init = json.load(file)
with open(vocab_new_path, "r") as file:
    vocab_new = json.load(file)

tokens_init_vocab = set(vocab_init.keys())
tokens_new_vocab = set(vocab_new.keys())
tokens_new_vocab = list(tokens_new_vocab)

id2token = {idx: token for idx, token in enumerate(tokens_new_vocab)}
token2id = {token: idx for idx, token in enumerate(tokens_new_vocab)}

with open("decoders/data/1M_vocab_embeddings/id2token.pkl", "wb") as id_file:
    pickle.dump(id2token, id_file)

with open("decoders/data/1M_vocab_embeddings/token2id.pkl", "wb") as token_file:
    pickle.dump(token2id, token_file)

# Obtain HN embeddings for the tokens in the 1M vocab
batch_size = 5000
num_batches = len(tokens_new_vocab) // batch_size + (
    1 if len(tokens_new_vocab) % batch_size != 0 else 0
)

all_predicted_input_embeddings = []
all_predicted_output_embeddings = []

target_surface_forms = get_surface_form_matrix(
    tokens_new_vocab,  # byte representation of the tokens to predict
    maxlen=hypernet.config.hn_surface_maxlen,
    tokenizer_to_use=hn_tokenizer,
)[0]
target_surface_forms = torch.from_numpy(target_surface_forms).to(device)
special_tokens_mask = torch.isin(
    target_surface_forms[:, 0],
    torch.tensor(hn_tokenizer.all_special_ids, device=device),
)

base_input_embeddings = base_model.get_input_embeddings().weight.data.to(device)
base_output_embeddings = base_model.get_output_embeddings().weight.data.to(device)

with torch.no_grad():
    for i in tqdm(range(num_batches), desc="Processing batches"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(tokens_new_vocab))

        batch_surface_forms = target_surface_forms[start_idx:end_idx]
        batch_special_tokens_mask = special_tokens_mask[start_idx:end_idx]

        predicted_input_embeddings, predicted_output_embeddings, _ = hypernet(
            batch_surface_forms,
            lang_index=lang_index,
            source_embeddings=source_embeddings,
        )

        predicted_input_embeddings[batch_special_tokens_mask] = base_input_embeddings[
            batch_surface_forms[batch_special_tokens_mask, 0]
        ]
        predicted_output_embeddings[batch_special_tokens_mask] = base_output_embeddings[
            batch_surface_forms[batch_special_tokens_mask, 0]
        ]

        all_predicted_input_embeddings.append(predicted_input_embeddings.cpu())
        all_predicted_output_embeddings.append(predicted_output_embeddings.cpu())

# Combine all embeddings
all_predicted_input_embeddings = torch.cat(all_predicted_input_embeddings, dim=0)
all_predicted_output_embeddings = torch.cat(all_predicted_output_embeddings, dim=0)

# Save the embeddings
torch.save(
    {
        "input_embeddings": all_predicted_input_embeddings,
        "output_embeddings": all_predicted_output_embeddings,
    },
    "decoders/data/1M_vocab_embeddings/large_HN_embeddings.pt",
)
