import torch
from collections import OrderedDict
from typing import Union


class LRU_Cache:
    def __init__(self, cache_size: int, emb_size: int = 768, device: str = "cpu"):
        self.capacity = cache_size
        assert str(device) == "cuda" or "cuda" in str(device)
        self.hypernet_preds = torch.zeros(cache_size, emb_size).to(device)
        self.biases = torch.zeros(cache_size, 1).to(device)
        self.token2idx = OrderedDict()
        self.free_indices = [i for i in range(cache_size - 1, -1, -1)]

    def get(self, key: str) -> Union[None, int]:
        if key not in self.cache:
            return None
        self.token2idx.move_to_end(key)
        return self.token2idx[key]

    def put(
        self, tokens: list, values: list, biases: torch.tensor = torch.tensor([])
    ) -> None:
        indices = []

        for token in tokens:
            if token in self.token2idx:
                self.token2idx.move_to_end(token)

            try:
                token_idx = self.free_indices.pop()
                self.token2idx[token] = token_idx
                indices.append(token_idx)
            except:
                raise Exception("No free indices available")

        self.hypernet_preds[indices] = values
        if biases.numel() > 0:
            self.biases[indices] = biases

    def evict_with_exceptions(
        self, tokens_eviction_exception: set, nr_tokens_to_evict: int
    ):
        evicted_tokens = 0
        tokens_to_remove = []
        iterated = 0
        for token, _ in self.token2idx.items():
            iterated += 1
            if token not in tokens_eviction_exception:
                self.free_indices.append(self.token2idx[token])
                tokens_to_remove.append(token)
                evicted_tokens += 1
                if evicted_tokens == nr_tokens_to_evict:
                    break
 
        for token in tokens_to_remove:
            del self.token2idx[token]

    def move_tokens_to_end(self, tokens):
        for token in tokens:
            try:
                self.token2idx.move_to_end(token)
            except:
                raise Exception(f"Token {token} not in cache.")

    @property
    def size(self) -> int:
        return len(self.token2idx)
