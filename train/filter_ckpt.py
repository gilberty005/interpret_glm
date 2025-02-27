from safetensors.torch import load_file, save_file

state_dict = load_file("last.safetensors")
sae_state_dict = {}

for key, value in state_dict.items():
    if key.startswith("sae_model."):
        new_key = key[len("sae_model."):]
        sae_state_dict[new_key] = value

print("SAE keys:", list(sae_state_dict.keys()))
save_file(sae_state_dict, "sae_fixed_checkpoint.safetensors")