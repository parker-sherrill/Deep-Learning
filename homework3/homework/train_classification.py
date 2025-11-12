import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.tensorboard as tb

from .datasets.classification_dataset import load_data
from .metrics import AccuracyMetric
from .models import Classifier, save_model


def train(
    exp_dir: str = "logs",
    model_name: str = "classifier",
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 2024,
    **kwargs,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Create log directory
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # Load model
    model = Classifier(**kwargs)
    model = model.to(device)
    model.train()

    # Load data with augmentation for training
    train_data = load_data(
        "classification_data/train",
        transform_pipeline="aug",  # Use augmentation for training
        shuffle=True,
        batch_size=batch_size,
        num_workers=2,
    )
    val_data = load_data(
        "classification_data/val",
        transform_pipeline="default",  # No augmentation for validation
        shuffle=False,
        batch_size=batch_size,
        num_workers=2,
    )

    # Create loss function and optimizer
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    global_step = 0
    best_val_acc = 0.0

    # Training loop
    for epoch in range(num_epoch):
        # Training phase
        model.train()
        train_metric = AccuracyMetric()
        train_metric.reset()

        for img, label in train_data:
            img, label = img.to(device), label.to(device)

            # Forward pass
            logits = model(img)
            loss = loss_func(logits, label)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Compute accuracy
            pred = logits.argmax(dim=1)
            train_metric.add(pred, label)

            # Log training loss
            logger.add_scalar("train/loss", loss.item(), global_step)
            global_step += 1

        # Validation phase
        with torch.inference_mode():
            model.eval()
            val_metric = AccuracyMetric()
            val_metric.reset()

            for img, label in val_data:
                img, label = img.to(device), label.to(device)

                # Forward pass
                logits = model(img)
                pred = model.predict(img)
                val_metric.add(pred, label)

        # Compute metrics
        train_acc = train_metric.compute()["accuracy"]
        val_acc = val_metric.compute()["accuracy"]

        # Log metrics
        logger.add_scalar("train/accuracy", train_acc, global_step)
        logger.add_scalar("val/accuracy", val_acc, global_step)

        # Print progress
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:3d} / {num_epoch:3d}: "
                f"train_acc={train_acc:.4f} "
                f"val_acc={val_acc:.4f}"
            )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model)
            print(f"  -> New best model saved! (val_acc={val_acc:.4f})")

    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved to {save_model(model)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2024)

    train(**vars(parser.parse_args()))

