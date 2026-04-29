import tomllib
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """Loads a TOML configuration file."""
    config_path_obj = Path(config_path)
    if not config_path_obj.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(config_path_obj, 'rb') as f:
        config = tomllib.load(f)
    return config

def get_model_config() -> Dict[str, Any]:
    """Loads the model configuration."""
    return load_config("configs/model_config.toml")

def get_data_config() -> Dict[str, Any]:
    """Loads the data configuration."""
    return load_config("configs/data_config.toml")

def get_training_config() -> Dict[str, Any]:
    """Loads the training configuration."""
    return load_config("configs/training_config.toml")

def get_testing_config() -> Dict[str, Any]:
    """Loads the testing configuration."""
    return load_config("configs/testing_config.toml")

def get_finetuning_config() -> Dict[str, Any]:
    """Loads the finetuning configuration."""
    return load_config("configs/finetuning_config.toml")


if __name__ == "__main__":
    # Example of loading and accessing configurations
    model_cfg = get_model_config()
    print("Model Configuration:", model_cfg)
    print("Vision Encoder:", model_cfg['vision']['encoder_name'])

    data_cfg = get_data_config()
    print("\nData Configuration:", data_cfg)
    print("Train Data Directory:", data_cfg['train_data_dir'])
    print("Use Language Modality:", data_cfg['modalities']['language'])

    train_cfg = get_training_config()
    print("\nTraining Configuration:", train_cfg)
    print("Max Epochs:", train_cfg['max_epochs'])
    print("Use Wandb:", train_cfg['use_wandb'])
    print("Finetuning Enabled:", train_cfg['finetuning']['enabled'])

    test_cfg = get_testing_config()
    print("\nTesting Configuration:", test_cfg)
    print("Checkpoint Path:", test_cfg['checkpoint_path'])

    finetune_cfg = get_finetuning_config()
    print("\nFinetuning Configuration:", finetune_cfg)
    print("Freeze Layers:", finetune_cfg.get('freeze_layers'))
