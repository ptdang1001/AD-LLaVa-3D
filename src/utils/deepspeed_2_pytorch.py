import os
from lightning.pytorch.utilities.deepspeed import (
    convert_zero_checkpoint_to_fp32_state_dict,
)

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"


def convert_checkpointFolder_to_fp32(deepspeed_checkpoint_folder):
    # Lightning deepspeed has saved a directory instead of a file
    # Convert the checkpoint to fp32
    input_deepspeed_checkpoint_dir_paht = deepspeed_checkpoint_folder
    output_fp32_checkpoint_file_path = os.path.join(
        deepspeed_checkpoint_folder, "pytorch_model.pt"
    )
    convert_zero_checkpoint_to_fp32_state_dict(
        input_deepspeed_checkpoint_dir_paht,
        output_fp32_checkpoint_file_path,
    )


def main():
    # Load the checkpoint folder
    deepspeed_checkpoint_path = "/path_to/deepspeed_checkpoint/"

    # Convert the checkpoint folder to fp32
    convert_checkpointFolder_to_fp32(deepspeed_checkpoint_path)

    print("Checkpoint conversion complete.")


if __name__ == "__main__":
    main()
