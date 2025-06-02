import torch

# === One-off file paths ===
full_ckpt_path = "model/checkpoints/alibi_model.pt"
output_path = "outputs/tuned_lens_heads/tuned_lens_heads.pt"

# === Load and extract ===
ckpt = torch.load(full_ckpt_path, map_location="cpu")
model_state = ckpt["model_state_dict"]

lens_state = {
    k.replace("tuned_lens_heads.", ""): v
    for k, v in model_state.items()
    if k.startswith("tuned_lens_heads.")
}

torch.save(lens_state, output_path)
print(f"✅ Tuned lens heads extracted and saved to {output_path}")
