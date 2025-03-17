import torch

def lora_projection(W_small, d_small=768, d_large=4096, r=64):
    """
    Use LoRA-style low-rank adaptation to project small model weights to a larger model
    :param W_small: Small model weights (768, 768)
    :param d_small: Small model hidden size
    :param d_large: Large model hidden size
    :param r: Low-rank dimension
    :return: Adapted large model weights (4096, 4096)
    """
    # Low-rank projection matrices
    A = torch.randn(r, d_small) * 0.02  # Shape: (64, 768)
    B = torch.randn(d_large, r) * 0.02  # Shape: (4096, 64)

    # Compute LoRA-style projection
    W_low = A @ W_small @ A.T  # Shape: (64, 64)
    W_large = B @ W_low @ B.T  # Shape: (4096, 4096)

    return W_large

def convert_to_larger_weights(gpt2_weights, d_small=768, d_large=4096):
    """
    Convert GPT2 extracted Transformer weights to LLaMA structure
    :param gpt2_weights: Extracted GPT2 weights
    :param d_small: GPT2 hidden size
    :param d_large: GPT-large hidden size
    :return: Adapted larger model weights
    """
    larger_model_weights = {}

    for key, W_small in gpt2_weights.items():
        if "attention" in key or "mlp" in key:  # Only convert key layers
            W_large = lora_projection(W_small, d_small, d_large)
            new_key = key.replace("gpt2_small", "gpt2_large")
            larger_model_weights[new_key] = W_large
        else:
            larger_model_weights[key] = W_small  # Directly copy other weights

    return larger_model_weights
