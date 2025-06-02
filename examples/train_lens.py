import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import os
import argparse
from model import get_model  
from utils.data_utils import load_and_prepare_data


@torch.no_grad()
def get_final_logits(model, xe):
    return model.lm_head(model.transformer.ln_f(xe))


def train_tuned_lens_heads(model, dataloader, device, epochs=3, lr=1e-4):
    # Freeze base model
    for param in model.parameters():
        param.requires_grad = False
    for head in model.tuned_lens_heads:
        for param in head.parameters():
            param.requires_grad = True

    optimizer = torch.optim.AdamW(model.tuned_lens_heads.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            xt, xe = model.transformer.embed(input_ids)
            final_logits = get_final_logits(model, xe).detach()

            lens_loss = 0
            for i, block in enumerate(model.transformer.h):
                xt, xe, *_ = block(xt, xe)
                xe_flat = xe.view(-1, xe.size(-1))
                pred_logits = model.tuned_lens_heads[i](xe_flat).view_as(final_logits)
                lens_loss += F.kl_div(
                    F.log_softmax(pred_logits, dim=-1),
                    F.softmax(final_logits, dim=-1),
                    reduction='batchmean'
                )

            optimizer.zero_grad()
            lens_loss.backward()
            optimizer.step()

            total_loss += lens_loss.item()

        print(f"[Epoch {epoch+1}] Tuned Lens KL loss: {total_loss / len(dataloader):.4f}")


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
    tokenizer = AutoTokenizer.from_pretrained("gpt2")  # or your tokenizer
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
    checkpoint = torch.load(args.model_ckpt)
    model = get_model("factored", config=checkpoint['config'])  # adjust model_type
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.to(args.device)

    train_tuned_lens_heads(model, dataloader, args.device, args.epochs, args.lr)

    # Save the tuned lens heads
    torch.save(model.tuned_lens_heads.state_dict(), os.path.join(args.output_dir, "tuned_lens_heads.pt"))
    print(f"Tuned Lens heads saved to {args.output_dir}")


if __name__ == "__main__":
    main()
