import os
import torch
import math
import time
from model import Transformer, DeepSeekConfig, device, device_type
from dataloader import DataLoaderLite
from inference import generate
    
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

def main():
    total_batch_size = 524288 # 2 ** 19 ~0.5M
    B = 32
    T = 256
    D = DeepSeekConfig.mtp_depth

    grad_accum_steps = total_batch_size // (B * T)

    # lower presicion for training
    torch.set_float32_matmul_precision('high')

    DeepSeek = Transformer(DeepSeekConfig)
    
    use_compile = False
    if use_compile:
        DeepSeek = torch.compile(DeepSeek)

    print(f"model size: {count_parameters(DeepSeek) // 1048576}M")

    train_loader = DataLoaderLite(B=B, T=T, D=D, split="train")
    val_loader = DataLoaderLite(B=B, T=T, D=D, split="val")

    max_lr = 6e-4
    min_lr = max_lr * 0.1
    warmup_steps = 715
    max_steps = 19073
    def get_lr(it):
        if it < warmup_steps:
            return max_lr * (it + 1) / warmup_steps
        
        if it > max_steps:
            return min_lr
        
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)

    optimizer = torch.optim.AdamW(DeepSeek.parameters(), lr=min_lr, betas=(0.9, 0.95), eps=1e-8)

    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"log.txt")
    with open(log_file, "w") as f: # open for writing to clear the file
        pass

    for step in range(max_steps):
        t0 = time.time()
        last_step = (step == max_steps - 1)

        if step % 250 == 0 or last_step:
            DeepSeek.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = 20
                for _ in range(val_loss_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, loss_main, loss_mtp = DeepSeek(x, 0, D, y)
                    loss_mtp = loss_mtp / val_loss_steps
                    loss_main = loss_main / val_loss_steps
                    val_loss_accum += loss_main.detach()
            print(f"validation loss: {val_loss_accum.item():.4f}")

            print("sampling: ")
            generate(DeepSeek, "I'm a language model which")

            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
            if step > 0 and (step % 3000 == 0 or last_step):
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    'model': DeepSeek.state_dict(),
                    'config': DeepSeek.config,
                    'step': step,
                    'val_loss': val_loss_accum.item()
                }
                torch.save(checkpoint, checkpoint_path)

        DeepSeek.train()
        optimizer.zero_grad()
        loss_accum = 0.0
        mtp_loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss_main, loss_mtp = DeepSeek(x, 0, D, y)
            loss_mtp = loss_mtp / grad_accum_steps
            loss_main = loss_main / grad_accum_steps
            loss_accum += loss_main.detach()
            mtp_loss_accum += loss_mtp.detach()
            loss_mtp.backward()
            loss_main.backward()

        norm = torch.nn.utils.clip_grad_norm_(DeepSeek.parameters(), 1)

        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.step()
        t1 = time.time()
        dt = (t1 - t0) * 1000
        tokens_processed = train_loader.B * train_loader.T * grad_accum_steps
        tokens_per_sec = tokens_processed / (t1 - t0)

        print(f"step {step:4d} | main loss: {loss_accum.item():.6f} | mtp loss: {mtp_loss_accum.item():.6f}| lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt:.4f}ms | tok/sec = {tokens_per_sec:.2f}")
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")

if __name__ == "__main__":
    main()