import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import os
import argparse
from model import get_model  
from utils.data_utils import load_and_prepare_data
from mytokenizers import create_tokenizer
from tqdm import tqdm
from accelerate import Accelerator
accelerator = Accelerator()

@torch.no_grad()
def get_final_logits(model, xe):
    """Project xe through final layernorm and LM head to get final logits."""
    return model.lm_head(model.transformer["ln_f"](xe))

def train_tuned_lens_heads(model, dataloader, optimizer, epochs=3):
    from tqdm import tqdm
    import torch.nn.functional as F

    # Freeze base model
    for param in model.parameters():
        param.requires_grad = False
    unwrapped_model = accelerator.unwrap_model(model)
    for head in unwrapped_model.tuned_lens_heads:
        for param in head.parameters():
            param.requires_grad = True

    model.train()

    for epoch in range(epochs):
        total_loss = 0
        with tqdm(dataloader, desc=f"Epoch {epoch+1}") as progress_bar:
            for batch_idx, batch in enumerate(progress_bar):
                input_ids = batch['input_ids']

                # Token + context streams
                tok_emb = model.transformer["wte"](input_ids)
                xt = model.transformer["drop"](tok_emb)
                xe = torch.zeros_like(xt)

                # === Forward pass to get final logits ===
                xt_final, xe_final = xt.clone(), xe.clone()
                for block in model.transformer.h:
                    xt_final, xe_final, _, _ = block(xt_final, xe_final, return_ffn_out=True)
                final_logits = model.lm_head(model.transformer["ln_f"](xe_final)).detach()

                # === Compute log-softmax of final logits for KL (P distribution) ===
                final_log_probs = F.log_softmax(final_logits, dim=-1)         # log P(x)
                final_probs = final_log_probs.exp()                           # P(x)

                if batch_idx == 0:
                    print("\n[DEBUG] Final logits stats:")
                    print(f"  shape: {final_logits.shape}")
                    print(f"  mean: {final_logits.mean().item():.4f}, std: {final_logits.std().item():.4f}")
                    print(f"  max: {final_logits.max().item():.4f}, min: {final_logits.min().item():.4f}")
                    print(f"  sample logits (first token): {final_logits[0,0,:5].tolist()}")

                lens_loss = 0
                for layer_idx, block in enumerate(model.transformer.h):
                    xt, xe, ffn_out, attn_out = block(xt, xe, return_ffn_out=True)

                    xe_flat = xe.view(-1, xe.size(-1))
                    pred_logits = model.tuned_lens_heads[layer_idx](xe_flat)
                    pred_logits = pred_logits.view_as(final_logits)

                    pred_log_probs = F.log_softmax(pred_logits, dim=-1)       # log Q(x)

                    # Stable KL: KL(P || Q) = sum P * (log P - log Q)
                    kl = torch.sum(final_probs * (final_log_probs - pred_log_probs), dim=-1).mean()

                    if batch_idx == 0:
                        print(f"\n[DEBUG] Layer {layer_idx} pred_logits stats:")
                        print(f"  shape: {pred_logits.shape}")
                        print(f"  mean: {pred_logits.mean().item():.4f}, std: {pred_logits.std().item():.4f}")
                        print(f"  sample logits (first token): {pred_logits[0,0,:5].tolist()}")
                        print(f"[DEBUG] Layer {layer_idx} KL: {kl.item():.4f}")

                    lens_loss += kl

                optimizer.zero_grad()
                accelerator.backward(lens_loss)
                torch.nn.utils.clip_grad_norm_(model.tuned_lens_heads.parameters(), max_norm=1.0)
                optimizer.step()

                progress_bar.set_postfix(kl_loss=lens_loss.item())
                total_loss += lens_loss.item()


        avg_loss = total_loss / len(dataloader)
        print(f"\n[Epoch {epoch+1}] Avg Tuned Lens KL loss: {avg_loss:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_ckpt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./trained_tuned_lens")
    parser.add_argument("--dataset", type=str, default="roneneldan/TinyStories")
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
 
    os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer and data
    tokenizer_params = {}
    tokenizer_params['use_fast'] = True
    tokenizer = create_tokenizer("gpt2", **tokenizer_params)

    dataloader, _ = load_and_prepare_data(
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        tokenizer=tokenizer,
        max_samples=50000,
        max_seq_length=128,
        batch_size=args.batch_size,
        split='train',
        shuffle=True
    )

    # Load model
    checkpoint = torch.load(args.model_ckpt, map_location="cpu", weights_only=False)
    model = get_model("factored", config=checkpoint["config"])  # adjust model_type
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    # Prepare optimizer before training
    optimizer = torch.optim.AdamW(model.tuned_lens_heads.parameters(), lr=args.lr)

    # Wrap model, optimizer, and dataloader with accelerate
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    train_tuned_lens_heads(model, dataloader, optimizer, args.epochs)

    # Save the tuned lens heads safely from the unwrapped model
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        torch.save(unwrapped_model.tuned_lens_heads.cpu().state_dict(), os.path.join(args.output_dir, "tuned_lens_heads.pt"))
        print(f"Tuned Lens heads saved to {args.output_dir}")


if __name__ == "__main__":
    main()
