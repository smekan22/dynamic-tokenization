import numpy as np
import scann
import torch

# Dictionary with 'input_embeddings' and 'output_embeddings' keys
embeddings = torch.load("decoders/data/1M_vocab_embeddings/large_HN_embeddings.pt")
normalized_dataset = (
    embeddings["output_embeddings"]
    / np.linalg.norm(embeddings["output_embeddings"], axis=1)[:, np.newaxis]
)

searcher = (
    scann.scann_ops_pybind.builder(normalized_dataset, 200, "dot_product")
    .tree(num_leaves=2000, num_leaves_to_search=250, training_sample_size=1_000_000)
    .score_ah(3, anisotropic_quantization_threshold=0.2)
    .reorder(200)
    .build()
)

INDEX_DIR = "decoders/data/scann_index/scann_index_6_neighbours_200_reorder_1000"
searcher.serialize(INDEX_DIR)
