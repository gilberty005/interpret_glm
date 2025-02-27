import torch
import os
from safetensors.torch import save_file, load_file

CHECKPOINT_DIR = "/burg/pmg/users/gy2322/results_l24_dim16384_k128/checkpoints"
OUTPUT_DIR = "/burg/pmg/users/gy2322/results_l24_dim16384_k128/safetensors"
os.makedirs(OUTPUT_DIR, exist_ok=True)

latest_ckpt_filename = "last.ckpt"
ckpt_path = os.path.join(CHECKPOINT_DIR, latest_ckpt_filename)
safetensors_path = os.path.join(OUTPUT_DIR, latest_ckpt_filename.replace(".ckpt", ".safetensors"))

def convert_ckpt_to_safetensors(ckpt_path, safetensors_path):
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint  

    # Clone shared tensors and ensure they are contiguous
    for key in state_dict.keys():
        if isinstance(state_dict[key], torch.Tensor):
            state_dict[key] = state_dict[key].clone().contiguous()
        else:
            state_dict[key] = state_dict[key]

    save_file(state_dict, safetensors_path)
    print(f"Saved SafeTensors model to: {safetensors_path}")

convert_ckpt_to_safetensors(ckpt_path, safetensors_path)

loaded_state_dict = load_file(safetensors_path)
print("\nVerification: SafeTensors file loaded successfully.")
print(f"Keys in SafeTensors model: {list(loaded_state_dict.keys())[:10]}")