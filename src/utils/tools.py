import os
import lightning as L
from rich.console import Console
from rich.table import Table
from rich.tree import Tree
from rich import box
import psutil
import platform
import torch
from typing import Union, Dict, List, Optional, Any, Tuple, Set
from collections import OrderedDict, defaultdict


RICH_AVAILABLE = True
CONSOLE = Console()


def handle_error(message: str, error: Exception, console: Console) -> None:
    """
    Enhanced error handler with detailed traceback and context.

    Args:
        message: Custom error message to display
        error: The exception that was caught
        console: Rich console instance for output
    """
    console.print(f"\n[bold red]{'=' * 80}[/bold red]")
    console.print(f"[bold red]{message}[/bold red]")
    console.print(f"[bold red]Type:[/bold red] {type(error).__name__}")
    console.print(f"[bold red]Error:[/bold red] {str(error)}")
    console.print("\n[bold red]Traceback:[/bold red]")
    console.print_exception(show_locals=True)
    console.print(f"[bold red]{'=' * 80}[/bold red]\n")


def set_and_print_seed(seed):
    """Set random seeds for reproducibility and display the information beautifully."""

    # Set the seed using Lightning
    L.seed_everything(seed)

    if not RICH_AVAILABLE:
        print(f"Random seed set to: {seed}")
        return seed

    # Create a nice table for seed information
    seed_table = Table(show_header=True, header_style="bold green", box=box.ROUNDED)
    seed_table.add_column("Component", style="dim", width=20)
    seed_table.add_column("Value", width=30)

    seed_table.add_row("Global Seed", f"{seed}")
    seed_table.add_row("PyTorch", "✓ Set")
    seed_table.add_row("NumPy", "✓ Set")
    seed_table.add_row("Python Random", "✓ Set")

    # Create a header
    CONSOLE.print(
        "\n[bold green]══════════════ Random Seed Configuration ══════════════[/bold green]"
    )
    CONSOLE.print(seed_table)
    CONSOLE.print(
        "[bold green]════════════════════════════════════════════════════════[/bold green]\n"
    )

    return seed


def print_device_info():
    """Display detailed, beautified information about the system and available computing devices."""
    if not RICH_AVAILABLE:
        # Fall back to simple output
        print("Device Information:")
        print(f"CPU: {os.cpu_count()} cores")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.device_count()} GPUs")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("GPU: 0 GPUs")
        print("\n")
        return

    # Create a header with a banner
    CONSOLE.print(
        "\n[bold blue]╔════════════════════════════════════════════════════╗[/bold blue]"
    )
    CONSOLE.print(
        "[bold blue]║             SYSTEM & DEVICE INFORMATION            ║[/bold blue]"
    )
    CONSOLE.print(
        "[bold blue]╚════════════════════════════════════════════════════╝[/bold blue]\n"
    )

    # System information
    sys_table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
    sys_table.add_column("System Component", style="dim", width=20)
    sys_table.add_column("Details", width=50)

    sys_table.add_row("OS", platform.platform())
    sys_table.add_row("Python", platform.python_version())
    sys_table.add_row("Architecture", platform.architecture()[0])

    # Add RAM information
    try:
        mem = psutil.virtual_memory()
        sys_table.add_row("Total RAM", f"{mem.total / (1024**3):.2f} GB")
        sys_table.add_row(
            "Available RAM",
            f"{mem.available / (1024**3):.2f} GB ({mem.available / mem.total * 100:.1f}%)",
        )
        sys_table.add_row("RAM Usage", f"{mem.percent}%")
    except AttributeError:
        sys_table.add_row("RAM Information", "Unable to retrieve")

    CONSOLE.print(sys_table)

    # CPU information
    cpu_table = Table(show_header=True, header_style="bold green", box=box.ROUNDED)
    cpu_table.add_column("CPU Attribute", style="dim", width=20)
    cpu_table.add_column("Value", width=50)

    cpu_table.add_row("CPU Model", platform.processor() or "Unknown")
    cpu_table.add_row("Logical Cores", f"{os.cpu_count()} cores")

    # Try to get physical cores
    try:
        physical_cores = psutil.cpu_count(logical=False)
        cpu_table.add_row("Physical Cores", f"{physical_cores} cores")
    except AttributeError:
        pass

    # Try to get CPU load
    try:
        cpu_table.add_row("Current Load", f"{psutil.cpu_percent()}%")
    except AttributeError:
        pass

    CONSOLE.print(cpu_table)

    # GPU information
    if torch.cuda.is_available():
        gpu_table = Table(show_header=True, header_style="bold yellow", box=box.ROUNDED)
        gpu_table.add_column("GPU Attribute", style="dim", width=20)
        gpu_table.add_column("Value", width=50)

        # CUDA information
        gpu_table.add_row("CUDA Version", torch.version.cuda or "Unknown")
        try:
            gpu_table.add_row(
                "cuDNN Version",
                str(torch.backends.cudnn.version())
                if torch.backends.cudnn.is_available()
                else "Not available",
            )
        except AttributeError:
            pass

        gpu_table.add_row("Number of GPUs", str(torch.cuda.device_count()))

        # Information for each GPU
        for i in range(torch.cuda.device_count()):
            gpu_table.add_section()
            gpu_table.add_row(f"[GPU {i}] Name", torch.cuda.get_device_name(i))

            # Get device properties
            try:
                props = torch.cuda.get_device_properties(i)
                gpu_table.add_row(f"[GPU {i}] Compute", f"{props.major}.{props.minor}")
                gpu_table.add_row(
                    f"[GPU {i}] Total Memory",
                    f"{props.total_memory / (1024**3):.2f} GB",
                )

                # Get memory stats
                mem_allocated = torch.cuda.memory_allocated(i) / (1024**3)
                mem_reserved = torch.cuda.memory_reserved(i) / (1024**3)
                gpu_table.add_row(
                    f"[GPU {i}] Memory Allocated", f"{mem_allocated:.2f} GB"
                )
                gpu_table.add_row(
                    f"[GPU {i}] Memory Reserved", f"{mem_reserved:.2f} GB"
                )
            except AttributeError:
                pass

        CONSOLE.print(gpu_table)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # For Apple Silicon (M1/M2) devices
        CONSOLE.print(
            "[bold yellow]MPS (Metal Performance Shaders) is available for Apple Silicon[/bold yellow]"
        )
    else:
        CONSOLE.print("[bold red]No GPU acceleration available[/bold red]")

    # PyTorch information
    torch_table = Table(show_header=True, header_style="bold cyan", box=box.ROUNDED)
    torch_table.add_column("PyTorch Attribute", style="dim", width=20)
    torch_table.add_column("Value", width=50)

    torch_table.add_row("PyTorch Version", torch.__version__)
    torch_table.add_row(
        "Default Device",
        str(
            torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                else "cpu"
            )
        ),
    )
    torch_table.add_row("Float32 MatMul", torch.get_float32_matmul_precision())

    CONSOLE.print(torch_table)

    # Add Lightning info
    lightning_table = Table(show_header=True, header_style="bold blue", box=box.ROUNDED)
    lightning_table.add_column("Lightning Attribute", style="dim", width=20)
    lightning_table.add_column("Value", width=50)

    lightning_table.add_row("Lightning Version", L.__version__)
    CONSOLE.print(lightning_table)

    # Final line
    CONSOLE.print(
        "\n[bold blue]══════════════ End of System Information ══════════════[/bold blue]\n"
    )


def print_model_info(
    model: Union[torch.nn.Module, L.LightningModule],
    detailed: bool = True,
    max_depth: int = 3,
    show_layers: bool = False,
    show_leaf_params: bool = False,
    title: str = "Model Information",
):
    """
    Display detailed model information with rich formatting.
    Optimized for Linux server environments with no graphical display.

    Args:
        model: PyTorch or Lightning model to analyze
        detailed: Whether to show detailed per-component breakdown
        max_depth: Maximum depth for module tree exploration
        show_layers: Whether to display individual layers
        show_leaf_params: Whether to show parameters for leaf modules
        title: Title for the main display panel
    """
    # Display header
    CONSOLE.print(f"\n[bold cyan]{'=' * 80}[/bold cyan]")
    CONSOLE.print(f"[bold cyan]{title.center(80)}[/bold cyan]")
    CONSOLE.print(f"[bold cyan]{'=' * 80}[/bold cyan]\n")

    # Get model class name and framework
    model_name = model.__class__.__name__
    is_lightning = isinstance(model, L.LightningModule)
    framework = "Lightning" if is_lightning else "PyTorch"

    # Collect basic statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    trainable_percent = (
        (trainable_params / total_params * 100) if total_params > 0 else 0
    )

    # Format parameter counts for display
    def format_params(count):
        if count >= 1e9:
            return f"{count / 1e9:.2f}B"
        elif count >= 1e6:
            return f"{count / 1e6:.2f}M"
        elif count >= 1e3:
            return f"{count / 1e3:.2f}K"
        else:
            return str(count)

    # Estimate model size in memory
    dtype_size = 4  # Default to float32
    for param in model.parameters():
        if param.dtype in (torch.float16, torch.bfloat16):
            dtype_size = 2
            break
    model_size_bytes = total_params * dtype_size

    def format_size(size_bytes):
        if size_bytes >= 1e9:
            return f"{size_bytes / 1e9:.2f} GB"
        elif size_bytes >= 1e6:
            return f"{size_bytes / 1e6:.2f} MB"
        elif size_bytes >= 1e3:
            return f"{size_bytes / 1e3:.2f} KB"
        else:
            return f"{size_bytes} B"

    # Collect device information
    devices = set()
    for param in model.parameters():
        devices.add(param.device)
    device_str = ", ".join(str(device) for device in devices)

    # Collect data type information
    dtypes = set()
    for param in model.parameters():
        dtypes.add(param.dtype)
    dtype_str = ", ".join(str(dtype).split(".")[-1] for dtype in dtypes)

    # Display model overview
    summary_table = Table(title=f"{model_name} Overview", box=box.ROUNDED)
    summary_table.add_column("Attribute", style="cyan")
    summary_table.add_column("Value", style="green")

    summary_table.add_row("Framework", framework)
    summary_table.add_row("Model Type", model_name)
    summary_table.add_row(
        "Total Parameters", f"{format_params(total_params)} ({total_params:,})"
    )
    summary_table.add_row(
        "Trainable Parameters",
        f"{format_params(trainable_params)} ({trainable_params:,}) [{trainable_percent:.2f}%]",
    )
    summary_table.add_row(
        "Non-Trainable Parameters",
        f"{format_params(non_trainable_params)} ({non_trainable_params:,}) [{100 - trainable_percent:.2f}%]",
    )
    summary_table.add_row("Model Size (Estimated)", format_size(model_size_bytes))
    summary_table.add_row("Device(s)", device_str)
    summary_table.add_row("Data Type(s)", dtype_str)

    # Calculate parameter statistics - FIXED to handle bfloat16
    if trainable_params > 0:
        try:
            import numpy as np

            # Stats for trainable parameters
            param_stats = {
                "mean": 0.0,
                "std": 0.0,
                "min": float("inf"),
                "max": float("-inf"),
                "median": 0.0,
                "q1": 0.0,
                "q3": 0.0,
                "count": 0,
            }

            # Use a safer approach that converts to float32 first
            for param in model.parameters():
                if param.requires_grad:
                    # Convert to float32 before converting to numpy
                    p_float = param.detach().float().cpu()
                    p_numpy = p_float.numpy().flatten()

                    # Update stats
                    if param_stats["count"] == 0:
                        param_stats["mean"] = np.mean(p_numpy)
                        param_stats["std"] = np.std(p_numpy)
                        param_stats["min"] = np.min(p_numpy)
                        param_stats["max"] = np.max(p_numpy)
                        q1, median, q3 = np.percentile(p_numpy, [25, 50, 75])
                        param_stats["q1"] = q1
                        param_stats["median"] = median
                        param_stats["q3"] = q3
                    else:
                        # Incremental update for mean and others
                        curr_count = param_stats["count"]
                        p_count = p_numpy.size
                        total_count = curr_count + p_count

                        # Update min/max
                        param_stats["min"] = min(param_stats["min"], np.min(p_numpy))
                        param_stats["max"] = max(param_stats["max"], np.max(p_numpy))

                        # For other stats, we use a simple approximation
                        # This is not perfectly accurate but avoids memory issues
                        param_stats["mean"] = (
                            param_stats["mean"] * curr_count + np.sum(p_numpy)
                        ) / total_count

                    param_stats["count"] += p_numpy.size

            if param_stats["count"] > 0:
                summary_table.add_row("Parameter Mean", f"{param_stats['mean']:.6f}")
                summary_table.add_row("Parameter Std", f"{param_stats['std']:.6f}")
                summary_table.add_row("Parameter Min", f"{param_stats['min']:.6f}")
                summary_table.add_row("Parameter Max", f"{param_stats['max']:.6f}")
                summary_table.add_row(
                    "Parameter Median", f"{param_stats['median']:.6f}"
                )
                summary_table.add_row(
                    "Parameter Q1/Q3",
                    f"{param_stats['q1']:.6f} / {param_stats['q3']:.6f}",
                )

        except Exception as e:
            CONSOLE.print(
                f"[yellow]Could not compute parameter statistics: {str(e)}[/yellow]"
            )

    CONSOLE.print(summary_table)

    # Function to recursively analyze model components
    def analyze_module(module, prefix="", depth=0):
        results = {}

        # Skip recursion if we've reached the maximum depth
        if depth > max_depth and not show_layers:
            return results

        for name, child in module.named_children():
            # Construct the full path to this module
            path = f"{prefix}.{name}" if prefix else name

            # Count parameters
            child_params = sum(p.numel() for p in child.parameters())
            child_trainable = sum(
                p.numel() for p in child.parameters() if p.requires_grad
            )
            child_nontrainable = child_params - child_trainable

            # Skip modules with no parameters if showing layers is disabled
            if child_params == 0 and not show_layers:
                continue

            # Store results
            results[path] = {
                "module": child,
                "params": child_params,
                "trainable": child_trainable,
                "nontrainable": child_nontrainable,
                "children": {},
            }

            # Recursively analyze children
            if len(list(child.named_children())) > 0:
                results[path]["children"] = analyze_module(child, path, depth + 1)

        return results

    # Analyze model structure if detailed view is requested
    if detailed:
        analysis = analyze_module(model)

        # Display hierarchical tree of components
        component_tree = Tree(f"[bold cyan]{model_name}[/bold cyan]")

        def add_to_tree(tree, analysis, parent_params=None):
            for name, data in analysis.items():
                # Get the module name (last part of the path)
                module_name = name.split(".")[-1]
                module_type = data["module"].__class__.__name__

                # Calculate percentage of total parameters
                percent_of_total = (
                    (data["params"] / total_params * 100) if total_params > 0 else 0
                )

                # Calculate percentage of parent parameters
                percent_of_parent = (
                    (data["params"] / parent_params * 100)
                    if parent_params and parent_params > 0
                    else 0
                )

                # Format the node label
                if data["params"] > 0:
                    param_info = f"[magenta]{format_params(data['params'])}[/magenta] "
                    param_info += f"([green]{format_params(data['trainable'])}[/green] trainable, "
                    param_info += f"[yellow]{format_params(data['nontrainable'])}[/yellow] frozen)"

                    if parent_params:
                        param_info += f" [blue]{percent_of_parent:.1f}%[/blue] of parent, [cyan]{percent_of_total:.1f}%[/cyan] of total"

                    node_label = (
                        f"[bold]{module_name}[/bold] ({module_type}): {param_info}"
                    )
                else:
                    node_label = f"[bold]{module_name}[/bold] ({module_type}): [grey]No parameters[/grey]"

                # Create node
                branch = tree.add(node_label)

                # Add leaf parameters if requested
                if show_leaf_params and len(data["children"]) == 0:
                    for param_name, param in data["module"].named_parameters(
                        recurse=False
                    ):
                        param_shape = "×".join(str(dim) for dim in param.shape)
                        param_count = param.numel()
                        requires_grad = "trainable" if param.requires_grad else "frozen"
                        leaf_node = f"[dim]{param_name}[/dim] [{param_shape}] {format_params(param_count)} [{requires_grad}]"
                        branch.add(leaf_node)

                # Recurse for children
                if data["children"]:
                    add_to_tree(branch, data["children"], data["params"])

        # Build tree
        add_to_tree(component_tree, analysis, total_params)
        CONSOLE.print(component_tree)

        # Display component table
        if show_layers:
            # Flatten the component hierarchy for tabular display
            flat_components = {}

            def flatten_hierarchy(analysis, prefix=""):
                for name, data in analysis.items():
                    full_path = name
                    flat_components[full_path] = data
                    if data["children"]:
                        flatten_hierarchy(data["children"], name)

            flatten_hierarchy(analysis)

            # Create component table
            component_table = Table(title="Component Analysis", box=box.ROUNDED)
            component_table.add_column("Component", style="cyan")
            component_table.add_column("Type", style="blue")
            component_table.add_column("Parameters", style="magenta", justify="right")
            component_table.add_column("Trainable", style="green", justify="right")
            component_table.add_column("% of Total", style="yellow", justify="right")

            # Sort components by parameter count
            sorted_components = sorted(
                flat_components.items(), key=lambda x: x[1]["params"], reverse=True
            )

            # Display top components in table
            max_rows = 20  # Limit to avoid overwhelming output
            for i, (name, data) in enumerate(sorted_components):
                if i >= max_rows:
                    component_table.add_row("...", "...", "...", "...", "...")
                    break

                module_type = data["module"].__class__.__name__
                total = format_params(data["params"])
                trainable = format_params(data["trainable"])
                percent = (
                    f"{data['params'] / total_params * 100:.2f}%"
                    if total_params > 0
                    else "0%"
                )

                component_table.add_row(name, module_type, total, trainable, percent)

            CONSOLE.print(component_table)

    # Group by module types
    if detailed:
        module_types = defaultdict(int)
        module_params = defaultdict(int)
        module_trainable = defaultdict(int)

        # Count parameters by module type
        for name, module in model.named_modules():
            module_type = module.__class__.__name__
            params = sum(p.numel() for p in module.parameters(recurse=False))
            trainable = sum(
                p.numel() for p in module.parameters(recurse=False) if p.requires_grad
            )

            if params > 0:  # Only count modules with parameters
                module_types[module_type] += 1
                module_params[module_type] += params
                module_trainable[module_type] += trainable

        # Display module type statistics
        if module_types:
            type_table = Table(title="Parameters by Module Type", box=box.ROUNDED)
            type_table.add_column("Module Type", style="cyan")
            type_table.add_column("Count", style="blue", justify="right")
            type_table.add_column("Parameters", style="magenta", justify="right")
            type_table.add_column("Trainable", style="green", justify="right")
            type_table.add_column("% Trainable", style="yellow", justify="right")
            type_table.add_column("% of Total", style="red", justify="right")

            # Sort by parameter count
            sorted_types = sorted(
                module_types.keys(), key=lambda x: module_params[x], reverse=True
            )

            for module_type in sorted_types:
                count = module_types[module_type]
                params = module_params[module_type]
                trainable = module_trainable[module_type]

                trainable_pct = (trainable / params * 100) if params > 0 else 0
                total_pct = (params / total_params * 100) if total_params > 0 else 0

                type_table.add_row(
                    module_type,
                    str(count),
                    format_params(params),
                    format_params(trainable),
                    f"{trainable_pct:.2f}%",
                    f"{total_pct:.2f}%",
                )

            CONSOLE.print(type_table)

    # Memory analysis
    memory_table = Table(title="Memory Analysis (Estimated)", box=box.ROUNDED)
    memory_table.add_column("Memory Type", style="cyan")
    memory_table.add_column("Size", style="green")
    memory_table.add_column("Notes", style="yellow")

    # Model parameters memory
    memory_table.add_row(
        "Parameters",
        format_size(model_size_bytes),
        f"Based on {total_params:,} parameters at {dtype_size} bytes each",
    )

    # Optimizer memory (Adam typically uses 4x parameter memory)
    adam_memory = model_size_bytes * 4
    memory_table.add_row(
        "Optimizer (Adam)",
        format_size(adam_memory),
        "Estimated for Adam (4x parameter memory)",
    )

    # Batch size memory (rough estimate based on input shape if available)
    batch_memory = "N/A"
    batch_note = "No example input available"
    try:
        if (
            hasattr(model, "example_input_array")
            and model.example_input_array is not None
        ):
            example_input = model.example_input_array
            if isinstance(example_input, torch.Tensor):
                input_size = example_input.element_size() * example_input.numel()
                input_shape = "×".join(str(dim) for dim in example_input.shape)
                batch_memory = format_size(input_size)
                batch_note = f"Based on input shape: {input_shape}"
    except:
        pass

    memory_table.add_row("Batch Input", batch_memory, batch_note)

    # Activations memory (rough estimate, typically 1-2x parameter memory)
    activation_memory = model_size_bytes * 1.5
    memory_table.add_row(
        "Activations (Forward)",
        format_size(activation_memory),
        "Estimated (1.5x parameter memory)",
    )

    # Gradients memory (same as parameter memory)
    memory_table.add_row(
        "Gradients (Backward)", format_size(model_size_bytes), "Same size as parameters"
    )

    # Total training memory
    total_memory = model_size_bytes + adam_memory + activation_memory + model_size_bytes
    memory_table.add_row(
        "Total for Training",
        format_size(total_memory),
        "Parameters + Optimizer + Activations + Gradients",
    )

    CONSOLE.print(memory_table)

    # Print sparsity information if available
    try:
        sparsity_table = Table(title="Parameter Sparsity Analysis", box=box.ROUNDED)
        sparsity_table.add_column("Metric", style="cyan")
        sparsity_table.add_column("Value", style="green")

        zero_count = 0
        total_count = 0

        for name, param in model.named_parameters():
            if param.requires_grad:
                # Convert to float before checking for zeros
                zeros = (param.detach().float() == 0).sum().item()
                total = param.numel()
                zero_count += zeros
                total_count += total

        if total_count > 0:
            sparsity = (zero_count / total_count) * 100
            sparsity_table.add_row("Zero Elements", f"{zero_count:,}")
            sparsity_table.add_row("Total Elements", f"{total_count:,}")
            sparsity_table.add_row("Sparsity", f"{sparsity:.2f}%")

            CONSOLE.print(sparsity_table)
    except Exception as e:
        CONSOLE.print(
            f"[yellow]Could not compute sparsity statistics: {str(e)}[/yellow]"
        )

    # Print footer
    CONSOLE.print(f"\n[bold cyan]{'=' * 80}[/bold cyan]")
    CONSOLE.print(f"[bold cyan]{'End of Model Information'.center(80)}[/bold cyan]")
    CONSOLE.print(f"[bold cyan]{'=' * 80}[/bold cyan]\n")
