from zett.utils import get_surface_form_matrix
import torch


def get_hn_embeddings_for_tokens(
    tokens: list,
    tokenizer,
    lang_index: int,
    hypernet,
    source_embeddings: torch.tensor,
    device,
    base_input_embeddings: torch.tensor,
    base_output_embeddings: torch.tensor,
):
    with torch.no_grad():
        target_surface_forms = get_surface_form_matrix(
            tokens,  # byte representation of the tokens to predict
            maxlen=hypernet.config.hn_surface_maxlen,
            tokenizer_to_use=tokenizer,
        )[0]
        target_surface_forms = torch.from_numpy(target_surface_forms).to(device)
        special_tokens_mask = torch.isin(
            target_surface_forms[:, 0],
            torch.tensor(tokenizer.all_special_ids, device=device),
        )

        predicted_input_embeddings, predicted_output_embeddings, _ = hypernet(
            target_surface_forms,
            lang_index=lang_index,
            source_embeddings=source_embeddings,
        )

        predicted_input_embeddings[special_tokens_mask] = base_input_embeddings[
            target_surface_forms[special_tokens_mask, 0]
        ]
        predicted_output_embeddings[special_tokens_mask] = base_output_embeddings[
            target_surface_forms[special_tokens_mask, 0]
        ]

        return predicted_input_embeddings.to(
            torch.bfloat16
        ), predicted_output_embeddings.to(torch.bfloat16)
