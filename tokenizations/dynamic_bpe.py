from collections import defaultdict, OrderedDict
from typing import Tuple, Dict
from datasets.formatting.formatting import LazyBatch
import string
from zett.utils import CHARS_TO_BYTES
from functools import lru_cache

from tokenizations.tokenizers_utils import tokenize, pretokenize
from tokenizers import pre_tokenizers
from collections import Counter


class Dynamic_BPE:
    def __init__(self, tokenizer, tokenizer_boundary: str = "pretokens"):
        self.tokenizer = tokenizer
        self.tokenizer_boundary = tokenizer_boundary
        self.special_token_map = set(tokenizer.special_tokens_map.values())
        self.punctuation_tokens = {
            token
            for token in tokenizer.vocab
            if any(c in token for c in """.,!?;:()-"'\/`$%&*+<=>@[]^_{|}~""")
            and not any(char.isdigit() for char in token)
        }
        self.debug = False

    @lru_cache(maxsize=None)
    def is_valid_pair(self, pair):
        token1, token2 = pair[0], pair[1]
        try:
            if self.tokenizer_boundary == "sentence":
                return (
                    token1 not in self.special_token_map
                    and token2 not in self.special_token_map
                )

            b = [CHARS_TO_BYTES[c] for c in token1 + token2]
            string = bytes(b).decode("utf-8")

            if self.tokenizer_boundary == "pretokens":
                return (
                    len(
                        self.tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(string)
                    )
                    == 1
                )
            elif self.tokenizer_boundary == "word":
                return (
                    len(pre_tokenizers.WhitespaceSplit().pre_tokenize_str(string)) == 1
                    and token1 not in self.special_token_map
                    and token2 not in self.special_token_map
                )
            elif self.tokenizer_boundary == "word_hyphen":
                cond1 = (
                    len(pre_tokenizers.WhitespaceSplit().pre_tokenize_str(string)) == 1
                    and token1 not in self.special_token_map
                    and token2 not in self.special_token_map
                )
                if (
                    cond1
                    and token1 not in self.punctuation_tokens
                    and token2 in self.punctuation_tokens
                ):
                    return token2 == "-"
                return cond1
        except UnicodeDecodeError:
            # this chunk of bytes is not a valid string, so we can't test it
            return True

    def get_most_frequent_pair(self, batch_tokens: Dict[str, int], check_valid=True):
        pair_freqs = Counter()
        for token_sequence in batch_tokens:
            pairs = zip(token_sequence, token_sequence[1:])
            if check_valid:
                pairs = (pair for pair in pairs if self.is_valid_pair(pair))
            pair_freqs.update(pairs)

        if pair_freqs:
            best_pair = max(pair_freqs, key=pair_freqs.get)
            return best_pair
        return ""

    def merge_pair(
        self, a: str, b: str, batch_tokens, ner: bool = False, batch_word_ids: list = []
    ):
        for idx, token_seq in enumerate(batch_tokens):
            i = 0
            new_token_seq = []
            new_word_ids = []
            token_seq = batch_tokens[idx]
            while i < len(token_seq):
                if (
                    i < len(token_seq) - 1
                    and token_seq[i] == a
                    and token_seq[i + 1] == b
                ):
                    new_token_seq.append(a + b)
                    if ner:
                        new_word_ids.append(batch_word_ids[idx][i])
                    i += 2
                else:
                    new_token_seq.append(token_seq[i])
                    if ner:
                        new_word_ids.append(batch_word_ids[idx][i])
                    i += 1
            batch_tokens[idx] = new_token_seq
            if ner:
                batch_word_ids[idx] = new_word_ids

        return batch_tokens, batch_word_ids

    def tokenize_base_case(
        self,
        batch_examples,
        max_length: int = 128,
        ner: bool = False,
    ):
        batch_tokens = []
        batch_word_tokens = []
        unique_tokens_original = set()
        batch_word_ids = []

        if ner:
            if isinstance(batch_examples, list):
                for batch_example in batch_examples:
                    tokens = ["<s>"]
                    word_ids = [None]
                    for word_index, word in enumerate(batch_example["tokens"]):
                        subtokens = self.tokenizer.tokenize(word, max_length=max_length)
                        tokens.extend(subtokens)
                        word_ids.extend([word_index] * len(subtokens))
                    if len(tokens) >= max_length:
                        tokens = tokens[: max_length - 1]
                        word_ids = word_ids[: max_length - 1]
                    tokens.append("</s>")
                    word_ids.append(None)

                    batch_tokens.append(tokens)
                    unique_tokens_original.update(tokens)
                    batch_word_ids.append(word_ids)
            else:
                for idx, _ in enumerate(batch_examples["tokens"]):
                    tokens = ["<s>"]
                    word_ids = [None]
                    for word_index, word in enumerate(batch_examples["tokens"][idx]):
                        subtokens = self.tokenizer.tokenize(word, max_length=max_length)
                        tokens.extend(subtokens)
                        word_ids.extend([word_index] * len(subtokens))
                    if len(tokens) >= max_length:
                        tokens = tokens[: max_length - 1]
                        word_ids = word_ids[: max_length - 1]
                    tokens.append("</s>")
                    word_ids.append(None)

                    unique_tokens_original.update(tokens)
                    batch_tokens.append(tokens)
                    batch_word_ids.append(word_ids)

        else:
            if isinstance(batch_examples, list):
                for batch_example in batch_examples:
                    tokens = (
                        ["<s>"]
                        + tokenize(batch_example["premise"], self.tokenizer)
                        + ["</s>", "</s>"]
                        + tokenize(batch_example["hypothesis"], self.tokenizer)
                        + ["</s>"]
                    )
                    batch_tokens.append(tokens)
                    unique_tokens_original.update(tokens)
                    if self.debug:
                        tokens_word = (
                            ["<s>"]
                            + pretokenize(batch_example["premise"], self.tokenizer)
                            + ["</s>", "</s>"]
                            + pretokenize(batch_example["hypothesis"], self.tokenizer)
                            + ["</s>"]
                        )
                        batch_word_tokens.append(tokens_word)
            else:
                for idx, _ in enumerate(batch_examples["premise"]):
                    tokens = (
                        ["<s>"]
                        + tokenize(batch_examples["premise"][idx], self.tokenizer)
                        + ["</s>", "</s>"]
                        + tokenize(batch_examples["hypothesis"][idx], self.tokenizer)
                        + ["</s>"]
                    )
                    batch_tokens.append(tokens)
                    unique_tokens_original.update(tokens)
                    if self.debug:
                        tokens_word = (
                            ["<s>"]
                            + pretokenize(
                                batch_examples["premise"][idx], self.tokenizer
                            )
                            + ["</s>", "</s>"]
                            + pretokenize(
                                batch_examples["hypothesis"][idx], self.tokenizer
                            )
                            + ["</s>"]
                        )
                        batch_word_tokens.append(tokens_word)
        return unique_tokens_original, batch_tokens, batch_word_tokens, batch_word_ids

    def initialize_position_tracking(self, batch_tokens):
        token_positions = defaultdict(list)
        for idx, tokens in enumerate(batch_tokens):
            for pos in range(len(tokens) - 1):
                pair = (tokens[pos], tokens[pos + 1])
                token_positions[pair].append((idx, pos))
        return token_positions

    def tokenize_batch(
        self,
        batch_examples: LazyBatch,
        max_nr_merges: int = 1000,
        max_length: int = 128,
        ner: bool = False,
    ):
        unique_tokens_bpe = set()
        batch_tokens = []
        batch_word_tokens = []
        batch_seq_lengths = []
        total_merges = 0

        _, batch_tokens, batch_word_tokens, batch_word_ids = (
            self.tokenize_base_case(
                batch_examples=batch_examples, max_length=max_length, ner=ner
            )
        )

        while total_merges < max_nr_merges:
            best_pair = self.get_most_frequent_pair(batch_tokens=batch_tokens)
            if best_pair == "":
                break

            total_merges += 1
            batch_tokens, batch_word_ids = self.merge_pair(
                a=best_pair[0],
                b=best_pair[1],
                batch_tokens=batch_tokens,
                ner=ner,
                batch_word_ids=batch_word_ids,
            )

        for _, tokenised_text in enumerate(batch_tokens):
            unique_tokens_bpe.update(tokenised_text)
            batch_seq_lengths.append(len(tokenised_text))

        if self.debug:
            for i in range(32):
                if i < len(batch_tokens) and batch_tokens[i] != batch_word_tokens[i]:
                    print(i)
                    print(batch_tokens[i])
                    print(batch_word_tokens[i])
                    
        return batch_tokens, unique_tokens_bpe, batch_seq_lengths, batch_word_ids