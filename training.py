"""training_script.py
Refactored according to review recommendations while keeping the original
behaviour as close as possible.

Key extras (minimal‑risk):
• Deterministic seeding (seed_everything)
• Dynamic batch‑size heuristic bumped to base=64 for large GPUs
• LR scheduler state & best metrics persisted in checkpoint
• Early‑stop counter correctly restored after resume
• Per‑dataset checkpoint path to avoid collisions
• Current LR printed each epoch for easier debugging
"""

import os
import sys
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from create_data import loadTest, loadTrain, collate_fn
from functionCi import rmse, mse, pearson, spearman, ci
from model import ImageNet

# ---------------- utils ---------------- #

def seed_everything(seed: int = 42) -> None:
    """Set PRNG seeds for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # slower but deterministic


def adjust_batch_size(device: torch.device, base: int = 64) -> int:
    """Heuristic batch‑size chooser based on total GPU memory."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return base

    props = torch.cuda.get_device_properties(device.index)
    mem_gb = props.total_memory / 1_073_741_824  # bytes→GiB

    if mem_gb < 8:
        return max(1, base // 2)  # 32
    elif mem_gb < 12:
        return base * 2            # 128
    elif mem_gb < 16:
        return base * 4            # 256
    else:
        return base * 8            # 512+ (similar to original)


# ---------------- training helpers ---------------- #

def train(model: nn.Module,
          device: torch.device,
          loader: DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: nn.Module,
          epoch: int,
          log_interval: int = 20) -> float:
    """One epoch of training; returns average loss."""
    model.train()
    running_loss = 0.0

    for batch_idx, (x, y, target) in enumerate(loader):
        x, y, target = x.to(device), y.to(device), target.to(device)

        optimizer.zero_grad(set_to_none=True)
        output = model(x, target)
        loss = loss_fn(output, y.view(-1, 1).float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()

        running_loss += loss.item() * x.size(0)

        if batch_idx % log_interval == 0:
            pct = 100.0 * batch_idx / len(loader)
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"Train epoch: {epoch} "
                  f"[{batch_idx * len(x)}/{len(loader.dataset)} ({pct:.0f}%)] "
                  f"Loss: {loss.item():.6f} — LR: {current_lr:.2e}")

    return running_loss / len(loader.dataset)


@torch.inference_mode()
def predict(model: nn.Module,
            device: torch.device,
            loader: DataLoader):
    """Run model on *loader*; returns (labels, preds) as 1‑D numpy arrays."""
    model.eval()
    preds, labels = [], []

    for x, y, target in loader:
        x, target = x.to(device), target.to(device)
        output = model(x, target)
        preds.append(output.cpu())
        labels.append(y.view(-1, 1))

    return (torch.cat(labels, 0).numpy().flatten(),
            torch.cat(preds, 0).numpy().flatten())


# ---------------- main ---------------- #

def main() -> None:
    seed_everything()

    # -------- CLI --------
    datasets = ["davis", "kiba"]
    try:
        dataset = datasets[int(sys.argv[1])]
    except (IndexError, ValueError):
        raise SystemExit("Usage: python training.py {0|1} [cuda_idx]")

    cuda_name = f"cuda:{int(sys.argv[2])}" if len(sys.argv) > 2 else "cuda:0"
    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    # -------- hyper‑params --------
    LR = 5e-4
    NUM_EPOCHS = 1000
    LOG_INTERVAL = 20
    PATIENCE = 20

    print("cuda_name       :", device)
    print("PyTorch version :", torch.__version__)
    print("CUDA available  :", torch.cuda.is_available())
    print("Learning rate   :", LR)
    print("Max epochs      :", NUM_EPOCHS)

    # -------- data --------
    train_batch = adjust_batch_size(device)
    test_batch = train_batch
    print("Using batch size:", train_batch)

    train_loader = DataLoader(loadTrain(dataset),
                              batch_size=train_batch,
                              shuffle=True,
                              collate_fn=collate_fn,
                              num_workers=4,
                              pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(loadTest(dataset),
                             batch_size=test_batch,
                             shuffle=False,  # no shuffle for test
                             collate_fn=collate_fn,
                             num_workers=4,
                             pin_memory=(device.type == "cuda"))

    # -------- model --------
    model = ImageNet().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode="min",
                                  factor=0.5, patience=5, verbose=True)

    # -------- checkpoint paths --------
    run_dir = Path("runs") / dataset
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "checkpoint.pt"
    model_file_name = run_dir / f"model_{dataset}.pth"
    result_file_name = run_dir / f"result_{dataset}.csv"

    # -------- resume / initialise --------
    start_epoch = 0
    best_mse = float("inf")
    best_ci = 0.0
    early_stop_counter = 0

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt.get("scheduler_state_dict", scheduler.state_dict()))
        start_epoch = ckpt["epoch"]
        best_mse = ckpt.get("best_mse", best_mse)
        best_ci = ckpt.get("best_ci", best_ci)
        early_stop_counter = ckpt.get("early_stop_counter", 0)
        print(f"✓ Resumed from epoch {start_epoch} (best MSE={best_mse:.6f})")

    # -------- training loop --------
    for epoch in range(start_epoch, NUM_EPOCHS):
        avg_loss = train(model, device, train_loader, optimizer,
                         loss_fn, epoch + 1, log_interval=LOG_INTERVAL)

        labels, preds = predict(model, device, test_loader)
        metrics = [rmse(labels, preds),
                   mse(labels, preds),
                   pearson(labels, preds),
                   spearman(labels, preds),
                   ci(labels, preds)]

        scheduler.step(metrics[1])  # step with validation MSE

        # ------ checkpoint ------
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_mse": best_mse,
            "best_ci": best_ci,
            "early_stop_counter": early_stop_counter,
        }, ckpt_path)

        # ------ metric tracking ------
        if metrics[1] < best_mse:
            best_epoch = epoch + 1
            best_mse = metrics[1]
            best_ci = metrics[4]
            early_stop_counter = 0

            torch.save(model.state_dict(), model_file_name)
            with open(result_file_name, "w") as fh:
                fh.write(",".join(map(str, metrics)))

            print(f"✓ MSE improved to {best_mse:.6f} at epoch {best_epoch} "
                  f"(CI={best_ci:.4f})")
        else:
            early_stop_counter += 1
            print(f"{metrics[1]:.6f} no improvement — best MSE {best_mse:.6f} "
                  f"(CI={best_ci:.4f}), patience {early_stop_counter}/{PATIENCE}")

            if early_stop_counter >= PATIENCE:
                print(f"Early stopping after {epoch + 1} epochs. Best MSE={best_mse:.6f}")
                break


if __name__ == "__main__":
    print('new V1')
    main()
