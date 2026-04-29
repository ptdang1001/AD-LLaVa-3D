# -*- utf-8 -*-

# built-in python modules
import json
import sys
import os
import time
import argparse

# third-party python modules
import wandb
import torch
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from rich.console import Console
from rich.panel import Panel
from rich.style import Style
from rich.progress import Progress, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.live import Live
from rich.traceback import install

# custom modules
# Import configuration loading utilities
sys.path.append(
    os.getcwd()
)  # Add current directory to sys.path to avoid src can not detected error.
from src.utils.load_configs import (
    get_model_config,
    get_data_config,
    get_training_config,
)
from src.utils.tools import (
    set_and_print_seed,
    print_device_info,
    print_model_info,
    handle_error,
)
from src.datamodules.utils import MRIDataModule
from src.models.multi_modal_model import LLaVANextVideoModel

# Set the environment variable to avoid tokenizers parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize the Rich Console
console = Console()
# Install rich traceback handler
# install(show_locals=True, width=120, word_wrap=True)

# set hardware
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("medium")

    # Create local Triton cache directory to avoid NFS warnings
    triton_cache_dir = os.path.join(
        "/path_to/cache/",
        f"triton_cache_{os.getuid()}_advlm",
    )
    print(f"Setting Triton cache directory to: {triton_cache_dir}")
    os.makedirs(triton_cache_dir, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = triton_cache_dir

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main(args):
    # set seed
    set_and_print_seed(42)

    # display device information
    print_device_info()

    # Define styles for rich printing
    info_style = Style(color="blue", bold=True)
    warning_style = Style(color="yellow", bold=True)
    success_style = Style(color="green", bold=True)
    error_style = Style(color="red", bold=True)

    # ------------------------
    # 1. Load Configurations
    # ------------------------
    console.print(
        Panel(f"[bold cyan]Loading Configurations[/bold cyan]", title="Setup")
    )
    model_cfg = get_model_config()
    data_cfg = get_data_config()
    train_cfg = get_training_config()

    console.print(f"  - [bold magenta]Training Config:[/bold magenta] {train_cfg}")
    console.print(f"  - [bold magenta]Model Config:[/bold magenta] {model_cfg}")
    console.print(f"  - [bold magenta]Data Config:[/bold magenta] {data_cfg}")

    # ------------------------
    # 2. Instantiate DataModule
    # ------------------------
    console.print(
        Panel("[bold cyan]Instantiating DataModule[/bold cyan]", title="Data")
    )
    try:
        # Initialize Data Module
        data_module = MRIDataModule(
            train_json=data_cfg["train_json"],
            test_json=data_cfg["test_json"],
            data_root=data_cfg["data_root"],
            num_workers=data_cfg.get("num_workers", 1),
            pin_memory=data_cfg.get("pin_memory", True),
            val_split=data_cfg.get("val_split", 0.25),
        )

        # Setup the datamodule
        data_module.setup()
        console.print(
            f"  - [bold green]DataModule instantiated successfully:[/bold green] {data_module.__class__.__name__}"
        )
        console.print("  - [bold green]Data setup complete.[/bold green]")

    except Exception as e:
        console.print(f"[bold red]Error instantiating DataModule:[/bold red] {e}")
        return

    # ------------------------
    # 3. Instantiate Model
    # ------------------------
    console.print(Panel("[bold cyan]Instantiating Model[/bold cyan]", title="Model"))
    try:
        model = LLaVANextVideoModel(lora_rank=args.lora_rank)
        console.print(
            f"  - [bold green]Model instantiated successfully:[/bold green] {model.__class__.__name__}"
        )
        # display model information
        print_model_info(model)
    except Exception as e:
        console.print(f"[bold red]Error instantiating Model:[/bold red] {e}")
        return

    # return True

    # ------------------------
    # 4. Define Callbacks
    # ------------------------

    # ------------------------
    # 5. Define Loggers
    # ------------------------
    console.print(Panel("[bold cyan]Defining Loggers[/bold cyan]", title="Logging"))

    # Wandb Logger (Optional)
    train_cfg["wandb"]["use_wandb"] = False
    if train_cfg["wandb"]["use_wandb"]:
        # current time
        current_time = time.strftime("%Y%m%d%H%M%S")
        wandb_project_name = "advlm"  # project name
        wandb_run_name = f"test_{current_time}"  # run name with timestamp
        wandb_log_dir = "./logs/"  # directory to save wandb logs
        # create the directory if it does not exist
        os.makedirs(wandb_log_dir, exist_ok=True)
        # set the wandb save directory
        wandb_logger = WandbLogger(
            save_dir=wandb_log_dir,
            project=wandb_project_name,
            name=wandb_run_name,
        )
        console.print(
            f"  - [bold green]WandbLogger added:[/bold green] Project: '{wandb_project_name}', Experiment: '{wandb_run_name}'"
        )
    else:
        console.print(
            "  - [yellow]Wandb logging is disabled in the configuration.[/yellow]"
        )

    # ------------------------
    # 6. Instantiate Trainer
    # ------------------------
    console.print(
        Panel(f"[bold cyan]Instantiating Trainer[/bold cyan]", title="Trainer")
    )
    try:
        trainer = L.Trainer(
            max_epochs=1,
            accelerator="gpu",  # 'cpu', 'gpu', 'tpu', 'ipu', 'hpu'
            devices=1,
            precision="bf16-mixed",  # '16', '32'
            enable_progress_bar=True,
        )
        console.print(
            "  - [bold green]Trainer instantiated successfully.[/bold green] Max Epochs: 1, Accelerator: GPU-CUDA, Devices: 1"
        )
    except Exception as e:
        handle_error(message="Error instantiating Trainer", error=e, console=console)
        return

    # Define model checkpoint map
    best_model_path = args.checkpoint_path
    console.print(f"[bold cyan]Using checkpoint:[/bold cyan] {best_model_path}")

    trainer.test(
        model=model,
        datamodule=data_module,
        ckpt_path=best_model_path,  # Load the best checkpoint
    )

    # get the result
    result = model.results_dict

    # save the result
    result_save_path = args.result_save_path
    # create the directory if it does not exist
    os.makedirs(os.path.dirname(result_save_path), exist_ok=True)
    with open(result_save_path, "w") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_rank", type=int, required=True, default=None)
    parser.add_argument("--checkpoint_path", type=str, required=True, default=None)
    parser.add_argument("--result_save_path", type=str, required=True, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
