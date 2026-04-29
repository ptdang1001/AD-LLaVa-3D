# AD-LLaVA-3D

A multimodal AI framework for Alzheimer's Disease (AD) analysis using 3D brain MRI. It fine-tunes [LLaVA-NeXT-Video-7B](https://huggingface.co/llava-hf/LLaVA-NeXT-Video-7B-hf) with LoRA adapters to answer clinical questions from 3D MRI volumes (ADNI / OASIS), combining a vision encoder (MAE-ViT), a language model (DeepSeek/LLaMA), and a gated MoE fusion module.

## Requirements

- Python ≥ 3.13.3
- CUDA-capable GPU
- [uv](https://docs.astral.sh/uv/) package manager

## Install

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and set up the project
git clone <repo-url>
cd AD-LLAVA-3D

# Install all dependencies
uv sync
```

## Configuration

Edit the TOML files under `configs/` before running:

| File | Purpose |
|---|---|
| `configs/data_config.toml` | Data paths (MRI H5 file, JSON splits) |
| `configs/model_config.toml` | Pretrained model directory |
| `configs/training_config.toml` | Training hyperparameters, W&B logging |
| `configs/testing_config.toml` | Checkpoint path for evaluation |

## Usage

### Train

```bash
uv run scripts/train.py
```

### Test

```bash
uv run scripts/test.py \
    --lora_rank 4 \
    --checkpoint_path /path/to/pytorch_model.pt \
    --result_save_path /path/to/test_result.json
```

Results are saved as a JSON file at `--result_save_path`.

## License

This project is licensed under the [CC BY-NC 4.0 License](LICENSE) — free for research and non-commercial use only.

## Project Structure

```
configs/        # TOML configuration files
scripts/        # Entry points: train.py, test.py, fine_tune.py
src/
  datamodules/  # MRI data loading and Lightning DataModule
  models/
    fusion/     # Gated MoE fusion module
    language/   # DeepSeek/LLaMA language model
    vision/     # MAE-ViT vision encoder
    multi_modal_model.py  # Main LLaVA-NeXT-Video Lightning module
  utils/        # Config loaders, tools
```
