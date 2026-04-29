import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Dict, Any, Union, Tuple
from peft import get_peft_model, LoraConfig, TaskType


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def DeepSeekLlama(
    pretrained_model_path: str = None,
    attn_strategy: str = "sdpa",
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map: str = DEVICE,
) -> torch.nn.Module:
    """
    Factory function to create a Llama model instance.

    Args:
        pretrained_model_path: Path to the pre-trained Llama model

    Returns:
        An instance of the Llama model
    """
    # Validate path
    if not os.path.exists(pretrained_model_path):
        raise FileNotFoundError(
            f"Pre-trained model path not found: {pretrained_model_path}"
        )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path, use_fast=True)
    # Ensure the tokenizer has a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    special_tokens = ["< |image| >", "< |table| >"]
    if special_tokens:
        # Create a dictionary for special tokens
        special_tokens_dict = {"additional_special_tokens": special_tokens}

        # Add the special tokens to the tokenizer
        num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)
        print(f"Added {num_added_tokens} special tokens: {special_tokens}")

        # Get the token IDs of the added special tokens
        for token in special_tokens:
            token_id = tokenizer.convert_tokens_to_ids(token)
            print(f"Special token '{token}' has ID: {token_id}")

    # Load pre-trained Llama model
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_path,
        attn_implementation=attn_strategy,  # Scaled dot-product attention for better performance
        torch_dtype=torch_dtype,
        device_map=device_map,
    )

    # Resize token embeddings to match the new vocabulary size
    # This is critical when adding new tokens!
    model.resize_token_embeddings(len(tokenizer))

    # Freeze all parameters
    if_freeze_model = True
    if if_freeze_model:
        for param in model.parameters():
            param.requires_grad = False
    else:
        for param in model.parameters():
            param.requires_grad = True

    # Apply LoRA if enabled
    use_lora = True
    lora_rank = 8
    lora_alpha = 16
    if use_lora:
        # Target attention projection layersls
        target_modules = [
            # Regular attention layers
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            # MLP layers that are often important
            "gate_proj",
            "up_proj",
            "down_proj",
            # For DeepSeek models that use attention variants
            "W_pack",
            "o_proj",  # Some DeepSeek models use packed weights
            # If you want to additionally target embedding layers
            # "embed_tokens",
            # "lm_head",
        ]
        print("Model layers that will be targeted by LoRA:")
        target_count = 0
        for name, module in model.named_modules():
            if any(target in name for target in target_modules):
                print(f"  - {name}: {type(module).__name__}")
                target_count += 1
        print(f"Total layers targeted: {target_count}")

        # Create LoRA config
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        # Apply LoRA to the model
        model = get_peft_model(model, lora_config)

        # Print trainable parameters info
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(
            f"LoRA enabled: {trainable_params:,} trainable parameters ({trainable_params / total_params:.2%} of total)"
        )

    return model, tokenizer
