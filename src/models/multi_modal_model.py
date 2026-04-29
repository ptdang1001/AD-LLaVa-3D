import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from typing import Dict, List, Optional, Tuple, Union, Any
import traceback
import sys
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.traceback import install, Traceback
from rich.table import Table
from rich.progress import track
import pysnooper
from peft import LoraConfig, get_peft_model, TaskType
from deepspeed.ops.adam import DeepSpeedCPUAdam


from transformers import (
    LlavaNextVideoForConditionalGeneration,
    LlavaNextVideoProcessor,
    # LlavaForConditionalGeneration,
    # LlavaProcessor,
    AutoProcessor,
    get_cosine_schedule_with_warmup,
    StoppingCriteria,
    StoppingCriteriaList,
    GenerationConfig,
)

# Install rich traceback handler for better error reporting
console = Console()


class StopOnTokens(StoppingCriteria):
    """Custom stopping criteria for generation"""

    def __init__(self, stop_token_ids):
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids, scores, **kwargs):
        """Check if any stop token has been generated"""
        for stop_ids in self.stop_token_ids:
            if input_ids[0][-len(stop_ids) :].tolist() == stop_ids:
                return True
        return False


class LLaVANextVideoModel(L.LightningModule):
    """LLaVA-Next-Video model with enhanced multi-modal support"""

    def __init__(
        self,
        lora_rank: int = 4,
        model_name: str = "llava-hf_LLaVA-NeXT-Video-7B-hf",
        lr: float = 1e-5,
        weight_decay: float = 1e-4,
        warmup_steps: int = 100,
        max_length: int = 20480,
    ):
        """
        Args:
            model_name: Hugging Face model name
            lr: Learning rate
            weight_decay: Weight decay
            warmup_steps: Number of warmup steps
            max_length: Maximum sequence length
        """
        try:
            super().__init__()
            self.save_hyperparameters()
            self.lora_rank = lora_rank

            # result
            self.results_dict = {}

            console.print(
                Panel(
                    "[bold blue]Initializing LLaVA-NeXT-Video Model[/]",
                    title="Model Setup",
                )
            )

            # Load model and processor
            pretrained_model_dir = "/path_to/Pre_Trained_Models/"
            pretrained_model_dir = os.path.join(pretrained_model_dir, model_name)

            console.print(f"[cyan]Loading model from:[/] {pretrained_model_dir}")

            # Set appropriate data type based on configuration
            dtype = torch.float16

            # Load model with error handling
            try:
                # Initialize model on CPU first
                self.model = LlavaNextVideoForConditionalGeneration.from_pretrained(
                    pretrained_model_dir,
                    torch_dtype=dtype,
                    load_in_8bit=False,
                    device_map=None,  # Let DeepSpeed handle device placement
                    attn_implementation="sdpa",
                    local_files_only=True,
                )
                # self.model.gradient_checkpointing_enable()

                # Apply LoRA if specified
                use_lora = True
                if use_lora:
                    lora_rank = self.lora_rank
                    lora_alpha = 2 * self.lora_rank
                    console.print(
                        f"[bold cyan]Applying LoRA fine-tuning:[/] rank={lora_rank}, alpha={lora_alpha}"
                    )

                    # Find all projection modules in the model
                    target_modules = [
                        "q_proj",
                        "k_proj",
                        "v_proj",
                        "o_proj",  # Attention projections
                        "gate_proj",
                        "up_proj",
                        "down_proj",  # MLP projections
                        "linear_1",
                        "linear_2",
                    ]

                    # Create LoRA configuration
                    lora_config = LoraConfig(
                        task_type=TaskType.CAUSAL_LM,
                        r=lora_rank,
                        lora_alpha=lora_alpha,
                        target_modules=target_modules,
                        lora_dropout=0.1,
                        bias="none",
                        inference_mode=False,
                    )

                    # Apply LoRA to the model
                    self.model = get_peft_model(self.model, lora_config)
                    self.model.print_trainable_parameters()

                # Initialize processor
                self.processor = LlavaNextVideoProcessor.from_pretrained(
                    pretrained_model_dir, local_files_only=True, use_fast=True
                )

                console.print(f"[green]✓ Model loaded successfully[/]")

            except Exception as e:
                console.print(f"[red]Error loading model:[/] {e}")
                raise e

        except Exception as e:
            console.print(f"[red]Error during model initialization:[/] {e}")
            raise e

    def _mask_prompt_from_labels(self, labels, prompt_no_answer, full_text):
        """
        Mask the tokens in labels that correspond to the prompt text by setting them to -100
        so they're ignored in the loss calculation.

        Args:
            labels: Tensor containing the full sequence labels [batch_size, seq_len]
            prompt_no_answer: The prompt text without the answer
            full_text: The full text input (prompt + answer)

        Returns:
            Masked labels tensor with prompt tokens set to -100
        """
        try:
            # Tokenize the prompt only (to find its length)
            prompt_tokens = self.processor(
                text=prompt_no_answer, return_tensors="pt", add_special_tokens=True
            ).to(self.device)

            prompt_length = prompt_tokens.input_ids.shape[1]

            # Create masked labels by setting prompt tokens to -100
            masked_labels = labels.clone()

            # Set all tokens before the answer to -100
            # Note: we mask up to prompt_length-1 to keep the last prompt token for better context
            # (This is a common practice to help with prediction of the first answer token)
            masked_labels[:, : prompt_length - 1] = -100

            # console.print(f"[dim]Masked {prompt_length - 1} prompt tokens in labels[/]")

            return masked_labels

        except Exception as e:
            console.print(f"[red]Error masking prompt tokens in labels:[/] {e}")
            raise e

    def save_lora_adapter(self, lora_save_dir):
        # save only LoRA adapter weights and processor
        self.model.save_pretrained(lora_save_dir)
        self.processor.save_pretrained(lora_save_dir)

    # @pysnooper.snoop()
    def forward(
        self,
        mri_data,  # Expects (D, 3, H, W) tensor
        prompt_with_answer,  # Full text prompt including answer (for training/validation) or just prompt (for testing)
    ):
        """
        Forward pass using LLaVA-Next-Video's native processing for a single 3D volume.

        Args:
            mri_data: Preprocessed volume tensor (D, 3, H, W)
            prompt_with_answer: Full text prompt including answer (for training/validation) or just prompt (for testing)

        Returns:
            Model outputs (including loss if labels are provided)
        """
        try:
            # Process inputs using the LLaVA-Next-Video processor
            # reshape the from_mri shape from [1,10,224,224] to [10,224,244] to [10,3,224,224]
            mri_data = mri_data.squeeze(0)
            mri_data = mri_data.unsqueeze(1)
            mri_data = mri_data.repeat(1, 3, 1, 1)

            # Convert to float32 for processor compatibility
            mri_data = mri_data.to(dtype=torch.float32)

            # Process inputs
            model_inputs = self.processor(
                text=prompt_with_answer,
                videos=mri_data,  # Pass the (D, 3, H, W) tensor directly
                return_tensors="pt",
                padding="longest",  # Use longest padding strategy
                truncation=True,
                max_length=10240,
                do_rescale=False,
            )
            model_inputs["labels"] = model_inputs["input_ids"].clone()

            # Move all inputs to the same device as the model's parameters
            device = next(self.model.parameters()).device
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

            # Forward through the model
            outputs = self.model(**model_inputs)

            return outputs

        except Exception as e:
            console.print(f"[red]Error in forward pass:[/] {e}")
            raise e

    def training_step(self, batch, batch_idx):
        """Training step for single 3D volume using native LLaVA-Next-Video processing"""
        try:
            prompt_no_answer = batch["prompt_no_answer"][0]
            prompt_no_answer = prompt_no_answer.replace("<image>", "<video>")
            answer = batch["answer"][0]
            answer = answer.replace("<|endofchunk|>", "</s>")
            mri_data = batch["from_mri"][0]

            # 4. Construct text input with answer
            prompt_with_answer = f"{prompt_no_answer} {answer}"

            # --- Forward Pass ---
            outputs = self.forward(
                mri_data=mri_data,
                prompt_with_answer=prompt_with_answer,
            )

            # Get the loss
            loss = outputs.loss

            # Log metrics
            self.log(
                "train_loss",
                loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                batch_size=1,  # Explicitly set batch size for logging
                sync_dist=True,
            )

            return loss

        except Exception as e:
            console.print(f"[red]Error in training step:[/] {e}")
            raise e

    def validation_step(self, batch, batch_idx):
        """Validation step for single 3D volume using native LLaVA-Next-Video processing"""
        try:
            prompt_no_answer = batch["prompt_no_answer"][0]
            prompt_no_answer = prompt_no_answer.replace("<image>", "<video>")
            answer = batch["answer"][0]
            answer = answer.replace("<|endofchunk|>", "</s>")
            mri_data = batch["from_mri"][0]

            # 4. Construct text input with answer
            prompt_with_answer = f"{prompt_no_answer} {answer}"
            # print(f"prompt_with_answer:\n{prompt_with_answer}")

            # --- Forward Pass ---
            outputs = self.forward(
                mri_data=mri_data, prompt_with_answer=prompt_with_answer
            )

            # Get the loss
            loss = outputs.loss

            # Log metrics
            self.log(
                "val_loss",
                loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                batch_size=1,  # Explicitly set batch size for logging
                sync_dist=True,
            )

            return loss

        except Exception as e:
            console.print(f"[red]Error in training step:[/] {e}")
            raise e

    def test_step(self, batch, batch_idx):
        """Test step with generation for single 3D volume using native LLaVA-Next-Video"""
        self.eval()  # Ensure model is in eval mode
        try:
            sample_id = batch["sample_id"][0]
            object_id = batch["object_id"][0]
            prompt_no_answer = batch["prompt_no_answer"][0]
            prompt_no_answer = prompt_no_answer.replace("<image>", "<video>")
            answer = batch["answer"][0]
            answer = answer.replace("<|endofchunk|>", "</s>")
            mri_data = batch["from_mri"][0]

            # Process inputs using the LLaVA-Next-Video processor
            # reshape the from_mri shape from [1,10,224,224] to [10,224,244] to [10,3,224,224]
            mri_data = mri_data.squeeze(0)
            mri_data = mri_data.unsqueeze(1)
            mri_data = mri_data.repeat(1, 3, 1, 1)

            # Convert to float32 for processor compatibility
            mri_data = mri_data.to(dtype=torch.float32)

            # --- Prepare Inputs for Generation ---
            # Use only the prompt for generation input text
            model_inputs = self.processor(
                text=prompt_no_answer,
                videos=mri_data,  # Pass the (D, 3, H, W) tensor
                return_tensors="pt",
                padding="longest",
                do_rescale=False,
            )

            # Move all inputs to the same device as the model's parameters
            device = next(self.model.parameters()).device
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

            # Define stopping criteria (optional, but good practice)
            stop_words = ["</s>", "\n\n", "Answer:"]  # Add relevant stop words
            stop_token_ids = []
            for word in stop_words:
                # Handle potential tokenization issues
                ids = self.processor.tokenizer.encode(word, add_special_tokens=False)
                if ids:
                    stop_token_ids.append(ids)

            stopping_criteria = (
                StoppingCriteriaList([StopOnTokens(stop_token_ids)])
                if stop_token_ids
                else None
            )

            # --- Generation ---
            with torch.no_grad():
                # Use model.generate for text generation
                generated_ids = self.model.generate(
                    **model_inputs,
                    # Generation parameters (adjust as needed)
                    num_beams=4,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    max_new_tokens=256,  # Increase max tokens if needed
                    stopping_criteria=stopping_criteria,
                    pad_token_id=self.processor.tokenizer.pad_token_id,  # Ensure pad token is set
                    eos_token_id=self.processor.tokenizer.eos_token_id,  # Ensure EOS token is set
                )

            # Decode generated text
            # generated_ids will contain input prompt + generated answer
            # Decode only the generated part by slicing based on input length
            input_token_len = model_inputs["input_ids"].shape[1]
            generated_text_ids = generated_ids[:, input_token_len:]
            generated_text = self.processor.batch_decode(
                generated_text_ids, skip_special_tokens=True
            )

            # Print the results for inspection
            console.print(f"\n--- Test Sample ---")
            console.print(f"[green]Sample ID:[/] {sample_id}")
            console.print(f"[green]Object ID:[/] {object_id}")
            console.print(f"[green]Prompt:[/] {prompt_no_answer}")
            console.print(f"[blue]Generated:[/]\n{generated_text[0].strip()}")
            if answer:
                console.print(f"[yellow]Expected Answer:[/] {answer}")
            console.print("-" * 100 + "\n")

            # save the result
            self.results_dict[sample_id] = {
                "sample_id": sample_id,
                "object_id": object_id,
                "prompt_no_answer": prompt_no_answer,
                "answer": answer,
                "generated_answer": generated_text[0].strip(),
            }
            print(f"self.results_dict: {self.results_dict}")
        except Exception as e:
            console.print(f"[red]Error in test step:[/] {e}")
            raise e

    def configure_optimizers(self):
        """Configure optimizers with weight decay and learning rate schedule"""
        try:
            trainable_params = filter(lambda p: p.requires_grad, self.parameters())
            optimizer = DeepSpeedCPUAdam(  # Changed from torch.optim.AdamW
                trainable_params,
                lr=self.hparams.lr,
                eps=1e-8,  # DeepSpeedCPUAdam default is 1e-8
                weight_decay=self.hparams.weight_decay,
                adamw_mode=True,  # Crucial for AdamW behavior with weight decay
            )
            return optimizer
        except Exception as e:
            console.print(f"[red]Error in configure_optimizers:[/] {e}")
            raise e
