import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from model import get_model
from utils.data_utils import load_and_prepare_data
from mytokenizers import create_tokenizer


@torch.no_grad()
def compute_layerwise_kl(model, dataloader, device, lens_heads=None):
    model.eval()

    if lens_heads:
        model.tuned_lens_heads.load_state_dict(torch.load(lens_heads, map_location=device))
        print("✅ Loaded tuned lens heads.")
    else:
        print("📌 Using untrained (random) lens heads.")

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)

        tok_emb = model.transformer["wte"](input_ids)
        xt = model.transformer["drop"](tok_emb)
        xe = torch.zeros_like(xt)

        # Final logits from full forward
        xt_f, xe_f = xt.clone(), xe.clone()
        for block in model.transformer.h:
            xt_f, xe_f, *_ = block(xt_f, xe_f, return_ffn_out=True)
        final_logits = model.lm_head(model.transformer["ln_f"](xe_f)).detach()
        final_log_probs = F.log_softmax(final_logits, dim=-1)
        final_probs = final_log_probs.exp()

        layer_kls = []
        for i, block in enumerate(model.transformer.h):
            xt, xe, *_ = block(xt, xe, return_ffn_out=True)
            xe_flat = xe.view(-1, xe.size(-1))
            pred_logits = model.tuned_lens_heads[i](xe_flat).view_as(final_logits)
            pred_log_probs = F.log_softmax(pred_logits, dim=-1)

            kl = torch.sum(final_probs * (final_log_probs - pred_log_probs), dim=-1).mean()
            layer_kls.append(kl.item())

        return layer_kls  # single batch


def main():
    model_ckpt = "model/checkpoints/alibi_model.pt"
    lens_ckpt = "outputs/tuned_lens_heads/tuned_lens_heads.pt"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = create_tokenizer("gpt2", use_fast=True)

    dataloader, _ = load_and_prepare_data(
        dataset_name="roneneldan/TinyStories",
        tokenizer=tokenizer,
        max_samples=1000,
        max_seq_length=128,
        batch_size=8,
        split='train',
        shuffle=False,
    )

    ckpt = torch.load(model_ckpt, map_location=device)
    model = get_model("factored", config=ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)

    print("\n🔍 Evaluating KL divergence...")
    kl_before = compute_layerwise_kl(model, dataloader, device)
    kl_after = compute_layerwise_kl(model, dataloader, device, lens_heads=lens_ckpt)

    print("\n📊 KL Divergence per Layer:")
    for i, (b, a) in enumerate(zip(kl_before, kl_after)):
        print(f"Layer {i:2d}: Before = {b:.4f}, After = {a:.4f}")

    plt.plot(kl_before, label="Before Training")
    plt.plot(kl_after, label="After Training")
    plt.xlabel("Layer")
    plt.ylabel("KL Divergence")
    plt.title("KL(P_final || Q_layer) Before vs After")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
