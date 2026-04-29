# -*- utf-8 -*-

# built-in python modules
import sys
import os
import time

# third-party python modules
import wandb
import torch
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DeepSpeedStrategy
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

    # Set PyTorch extensions directory
    torch_extensions_dir = os.path.join(
        "/path_to/cache/torch_extensions/",
        "torch_extensions",
    )
    os.makedirs(torch_extensions_dir, exist_ok=True)
    os.environ["TORCH_EXTENSIONS_DIR"] = torch_extensions_dir

    # Set environment variable to avoid memory fragmentation
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
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
            num_workers=data_cfg.get("num_workers", 2),
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
        model = LLaVANextVideoModel()
        console.print(
            f"  - [bold green]Model instantiated successfully:[/bold green] {model.__class__.__name__}"
        )
        # display model information
        print_model_info(model)
    except Exception as e:
        console.print(f"[bold red]Error instantiating Model:[/bold red] {e}")
        return

    # ------------------------
    # 4. Define Callbacks
    # ------------------------
    console.print(Panel("[bold cyan]Defining Callbacks[/bold cyan]", title="Callbacks"))
    callbacks = []

    data_version = data_cfg["data_root"].split("/")[-2]
    checkpoint_save_dir = "path_to/checkpoints"
    # Create the directory if it does not exist
    os.makedirs(checkpoint_save_dir, exist_ok=True)
    # Model Checkpoint Callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_save_dir,
        save_weights_only=True,
        monitor="val_loss",  # Monitor validation loss
        mode="min",  # 'min' or 'max'
        filename="advlm-{epoch:03d}-{val_loss:.4f}",  # Adjust filename as needed
        save_top_k=1,  # Save the top k best models
        auto_insert_metric_name=False,  # Avoid auto-inserting metric name in filename
        verbose=True,  # Verbose logging
    )
    callbacks.append(checkpoint_callback)
    console.print(
        f"  - [bold green]ModelCheckpoint callback added:[/bold green] Saving to '{train_cfg['checkpoint_dir']}/best'"
    )

    # Early Stopping Callback (Optional)
    if train_cfg["use_early_stopping"]:
        early_stopping_callback = EarlyStopping(
            monitor="val_loss",  # Monitor validation loss
            patience=3,
            mode="min",  # 'min' or 'max'
            verbose=True,
        )
        callbacks.append(early_stopping_callback)
        console.print(
            "  - [bold green]EarlyStopping callback added:[/bold green] Monitor: 'val_loss', Patience: 3, Mode: 'min'"
        )
    else:
        console.print(
            "  - [yellow]EarlyStopping is disabled in the configuration.[/yellow]"
        )

    # ------------------------
    # 5. Define Loggers
    # ------------------------
    console.print(Panel("[bold cyan]Defining Loggers[/bold cyan]", title="Logging"))

    # Wandb Logger (Optional)
    if train_cfg["wandb"]["use_wandb"]:
        # current time
        current_time = time.strftime("%Y%m%d%H%M%S")
        wandb_project_name = "ADVLM"  # project name
        wandb_run_name = f"train_{current_time}"  # run name with timestamp
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
        deepspeed_strategy = DeepSpeedStrategy(
            stage=3,
            offload_optimizer=True,
            offload_parameters=False,
        )
        trainer = L.Trainer(
            max_epochs=train_cfg["trainer"]["max_epochs"],
            accelerator="auto",  # 'cpu', 'gpu', 'tpu', 'ipu', 'hpu'
            devices="auto",
            callbacks=callbacks,
            strategy=deepspeed_strategy,
            logger=wandb_logger if train_cfg["wandb"]["use_wandb"] else None,
            precision="16-mixed",  # '16', '32'
            gradient_clip_val=1.0,
            accumulate_grad_batches=8,
            enable_progress_bar=True,
        )
        console.print(
            f"  - [bold green]Trainer instantiated successfully.[/bold green] Max Epochs: {train_cfg['trainer']['max_epochs']}, Accelerator: GPU-CUDA, Devices: 1"
        )
    except Exception as e:
        handle_error(message="Error instantiating Trainer", error=e, console=console)
        return

    # ------------------------
    # 7. Training with Live Display
    # ------------------------
    console.print(
        Panel(
            "[bold cyan]Starting Training ...[/bold cyan]",
            title="Training",
        )
    )
    trainer.fit(
        model=model,
        datamodule=data_module,
    )
    # save the lora adapter
    lora_save_dir = os.path.join(checkpoint_save_dir, "lora_adapter")
    os.makedirs(lora_save_dir, exist_ok=True)
    model.save_lora_adapter(lora_save_dir)
    # print the lora adapter path
    console.print(
        f"  - [bold green]LoRA adapter saved to:[/bold green] {lora_save_dir}"
    )
    console.print(
        Panel("[bold green]Training Finished![/bold green]", title="Training")
    )
    # Print the best checkpoint path
    best_model_path = checkpoint_callback.best_model_path
    console.print(
        f"  - [bold green]Best model saved at:[/bold green] {best_model_path}"
    )
    if train_cfg["wandb"]["use_wandb"]:
        wandb.finish()  # Finish the wandb run
    console.print(
        Panel("[bold green]Wandb run finished![/bold green]", title="Logging")
    )


if __name__ == "__main__":
    main()
