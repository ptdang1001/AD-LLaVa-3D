import os
import torch
from transformers import ViTMAEForPreTraining, AutoImageProcessor
from typing import Optional, Dict, Any, Union, Tuple
from peft import get_peft_model, LoraConfig, TaskType

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def MaeViT(
    pretrained_model_path: str = None,
    mask_ratio: float = 0.75,
    attn_strategy: str = "sdpa",
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map: str = DEVICE,
) -> Tuple[ViTMAEForPreTraining, AutoImageProcessor]:
    """
    Load the MAE ViT model and image processor.

    Args:
        pretrained_model_path: Path to the pre-trained MAE ViT model
        mask_ratio: Ratio of patches to mask during training (0.0-1.0)
        torch_dtype: Precision for model computation
        map_location: Device to load the model on
    Returns:
        Tuple containing the MAE ViT model and image processor
    """
    # if file not exists, arrise an error
    if not os.path.exists(pretrained_model_path):
        raise FileNotFoundError(f"Model path {pretrained_model_path} does not exist.")

    image_processor = AutoImageProcessor.from_pretrained(
        pretrained_model_path, use_fast=True
    )

    model = ViTMAEForPreTraining.from_pretrained(
        pretrained_model_path,
        attn_implementation=attn_strategy,  # Scaled dot-product attention for better performance
        torch_dtype=torch_dtype,
        mask_ratio=mask_ratio,
        device_map=device_map,
    )

    # freeze the model parameters
    if_freeze_model = False
    if if_freeze_model:
        for param in model.parameters():
            param.requires_grad = False

    use_lora = False
    lora_rank = 16
    lora_alpha = 32
    if use_lora:
        config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=["query", "key", "value", "dense"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        model = get_peft_model(model, config)

        # Print trainable parameters info
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(
            f"LoRA enabled: {trainable_params:,} trainable parameters ({trainable_params / total_params:.2%} of total)"
        )

    return model, image_processor
