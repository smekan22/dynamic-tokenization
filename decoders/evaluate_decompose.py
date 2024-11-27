from transformers import HfArgumentParser, AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass
import torch
import hashlib
import regex as re
from tqdm.auto import tqdm
from datasets import load_dataset
import pandas as pd
import json

TOKENIZER = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
FORBIDDEN_WORDS = {"child", "kid", "girl", "boy", "man", "woman", "fuck", "shit"}
WORDLIST_PATH = "data/words_alpha.txt"  # from https://github.com/dwyl/english-words/blob/master/words_alpha.txt
EVAL_CUTOFF = 1000


def process(
    input_ids: torch.tensor,
    decompose_start_replace_id: int = 128254,
    decompose_end_replace_id: int = 128255,
) -> torch.tensor:
    decompose = False
    decomposed_input_ids = []

    for input_id in input_ids:
        if input_id == decompose_start_replace_id:
            decompose = True
            decomposed_input_ids.append(input_id)
        elif input_id == decompose_end_replace_id:
            decompose = False
            decomposed_input_ids.append(input_id)
        elif decompose:
            for byte in TOKENIZER.convert_ids_to_tokens(input_id):
                decomposed_input_ids.append(TOKENIZER.convert_tokens_to_ids(byte))
        else:
            decomposed_input_ids.append(input_id)

    return decomposed_input_ids


def prompt_to_tokens(prompt: str, do_decompose: bool) -> list:
    if do_decompose:
        system_prompt = "You are a helpful assistant specialized in solving character-knowledge intensive tasks by decomposing the relevant subwords into individual characters or bytes."
    else:
        system_prompt = "You are a helpful assistant specialized in solving character-knowledge intensive tasks."

    if do_decompose:
        proc_prompt = prompt.replace(
            "<decompose>", "<|reserved_special_token_246|>"
        ).replace("</decompose>", "<|reserved_special_token_247|>")
    else:
        proc_prompt = prompt.replace("<decompose>", "").replace("</decompose>", "")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": proc_prompt},
    ]

    tokens = process(
        TOKENIZER.apply_chat_template(messages, add_generation_prompt=True)
    )
    return tokens


class ReverseEval:
    def __init__(self, fewshot_words: list):
        self.fewshot_words = fewshot_words

    def get_prompt(self, word, n_fewshot=0):
        prompt = ""

        if n_fewshot > 0:
            prompt = "Here are examples of words and their reversals."
            for i in range(n_fewshot):
                prompt += f"\n\nWord: <decompose>{self.fewshot_words[i]}</decompose>\nReversed: <decompose>{self.fewshot_words[i][::-1]}</decompose>"

            prompt += "\n\n"

        prompt += f"Reverse the word '<decompose>{word}</decompose>'."
        return prompt

    def parse_response(self, response_tokens: list, word: str):
        return {
            "acc": any(
                chunk == word[::-1]
                for chunk in re.split("\W+", TOKENIZER.decode(response_tokens))
            )
        }


class PrefixSuffixEval:
    def __init__(self, fewshot_words: list, kind: str, max_n_letters: int):
        self.fewshot_words = fewshot_words
        self.kind = kind
        self.max_n_letters = max_n_letters

    def to_starfix(self, word: str, n_letters: int) -> str:
        if self.kind == "prefix":
            return word[:n_letters]
        elif self.kind == "suffix":
            return word[-n_letters:]

    def get_prompt(self, word: str, n_fewshot: int = 0) -> str:
        n_letters = (
            int(hashlib.sha256(word.encode("utf-8")).hexdigest(), 16)
            % self.max_n_letters
            + 1
        )
        while n_letters > len(word):
            n_letters -= 1

        prompt = ""
        firstlast = "first" if self.kind == "prefix" else "last"
        letter_or_letters = "letter" if n_letters == 1 else "letters"

        if n_fewshot > 0:
            prompt = f"Here are examples of words and their {firstlast} {n_letters} {letter_or_letters}."
            starfix = self.to_starfix(word, n_letters)

            for i in range(n_fewshot):
                prompt += (
                    f"\n\nWord: {self.fewshot_words[i]}\n{self.kind.title()}: {starfix}"
                )

            prompt += "\n\n"

        prompt += f"What is the string of the {firstlast} {n_letters} {letter_or_letters} in '<decompose>{word}</decompose>'?"
        return prompt

    def parse_response(self, response_tokens: list, word: str) -> dict:
        response_string = TOKENIZER.decode(response_tokens)

        firstlast = "first" if self.kind == "prefix" else "last"
        re_match = re.search(
            rf"What is the string of the {firstlast} (\d+) letter", response_string
        )
        n_letters = int(re_match.group(1))

        starfix = self.to_starfix(word, n_letters)
        return {
            "acc": any(
                chunk == starfix
                for chunk in re.split("\W+", TOKENIZER.decode(response_tokens))
            ),
            "n_letters": n_letters,
        }


class CompoundSplitEval:
    def __init__(self, fewshot_words: list, compoundpiece_df: pd.DataFrame):
        self.fewshot_words = fewshot_words
        self.word_to_segmentation = {
            row["word"]: row["segmentation"] for i, row in compoundpiece_df.iterrows()
        }

    def get_prompt(self, word: str, n_fewshot: int = 0) -> str:
        prompt = ""

        if n_fewshot > 0:
            prompt = "Here are examples of words and their compound parts."

            for i in range(n_fewshot):
                fewshot_word = self.fewshot_words[i]
                fewshot_parts = [
                    "<decompose>" + x + "</decompose>"
                    for x in self.word_to_segmentation[fewshot_word].split("-")
                ]

                prompt += f"\n\nWord: <decompose>{fewshot_word}</decompose>\nParts: {', '.join(fewshot_parts)}"

            prompt += "\n\n"

        prompt += f"Which compound parts is '<decompose>{word}</decompose>' made up of?"
        return prompt

    def parse_response(self, response_tokens: list, word: str) -> dict:
        parts = self.word_to_segmentation[word].split("-")
        chunks_in_parts = set(re.split("\W+", TOKENIZER.decode(response_tokens))) & set(
            parts
        )

        return {
            "acc": len(chunks_in_parts) >= len(parts) - 1,
            "parts": parts,
            "chunks_in_parts": chunks_in_parts,
        }


def evaluate(
    cls: ReverseEval | PrefixSuffixEval | CompoundSplitEval,
    eval_words: list,
    model,
    do_decompose: bool = True,
    n: int = None,
    n_fewshot: int = 0,
    zs_cot: bool = False,
    max_new_tokens: int = 64,
) -> list:
    words_to_use = eval_words if n is None else eval_words[:n]
    results = []

    for word in tqdm(words_to_use):
        prompt = cls.get_prompt(word, n_fewshot=n_fewshot)
        if zs_cot:
            prompt += " Explain your reasoning and think step by step."
        tokens = prompt_to_tokens(prompt, do_decompose=do_decompose)
        tokens = torch.tensor([tokens])
        out = model.generate(
            input_ids=tokens,
            attention_mask=torch.ones_like(tokens),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=TOKENIZER.eos_token_id,
        )[0]
        results.append((out, cls.parse_response(out, word)))

    return results


@dataclass
class Args:
    model_name: str = "cambridgeltl/Llama-3.2-1B-Instruct"
    eval: str = "reverse"  # or "prefix" or "suffix" or "compoundsplit"
    eval_kwargs: str | None = None  # e.g. "{\"max_n_letters\": 5}" for prefix/suffix
    base_model_name: str = (
        "meta-llama/Llama-3.2-1B-Instruct"  # only needed for LoRA interp
    )
    lora_interp: float | None = None
    limit: int | None = None
    n_fewshot: int = 0
    zs_cot: bool = False


if __name__ == "__main__":
    (args,) = HfArgumentParser([Args]).parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    if args.lora_interp is not None:
        base_model = AutoModelForCausalLM.from_pretrained(args.base_model_name)
        state_dict = model.state_dict()

        for (name1, param_original), (name2, param_ft) in zip(
            base_model.named_parameters(), model.named_parameters()
        ):
            assert name1 == name2
            name = name1
            state_dict[name].data[:] = (
                param_original + (param_ft - param_original) * args.lora_interp
            )

    compoundpiece_dset = load_dataset(
        "benjamin/compoundpiece", "wiktionary", split="train"
    )
    compoundpiece_df = pd.DataFrame(compoundpiece_dset)

    wordlist_words = {x.strip() for x in open(WORDLIST_PATH)}

    words = compoundpiece_df[compoundpiece_df["lang"] == "en"]["word"].tolist()
    words = [
        word
        for word in words
        if word not in wordlist_words
        and all(c.isalpha() for c in word)
        and not any(x in word for x in FORBIDDEN_WORDS)
    ]
    eval_words = words[:EVAL_CUTOFF]
    fewshot_words = words[EVAL_CUTOFF:]

    if args.eval_kwargs is None:
        eval_kwargs = {}
    else:
        eval_kwargs = json.loads(args.eval_kwargs)

    if args.eval == "reverse":
        eval_cls = ReverseEval(fewshot_words, **eval_kwargs)
    elif args.eval == "prefix":
        eval_cls = PrefixSuffixEval(fewshot_words, "prefix", **eval_kwargs)
    elif args.eval == "suffix":
        eval_cls = PrefixSuffixEval(fewshot_words, "suffix", **eval_kwargs)
    elif args.eval == "compoundsplit":
        eval_cls = CompoundSplitEval(fewshot_words, compoundpiece_df, **eval_kwargs)

    results = evaluate(
        eval_cls,
        eval_words,
        model,
        n=args.limit,
        n_fewshot=args.n_fewshot,
        zs_cot=args.zs_cot,
    )
