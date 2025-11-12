import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.tensorboard as tb

from .datasets.road_dataset import load_data
from .metrics import DetectionMetric
from .models import Detector, save_model


def train(
    exp_dir: str = "logs",
    model_name: str = "detector",
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 32,
    seed: int = 2024,
    seg_weight: float = 1.0,
    depth_weight: float = 1.0,
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
    model = Detector(**kwargs)
    model = model.to(device)
    model.train()

    # Load data
    train_data = load_data(
        "drive_data/train",
        transform_pipeline="default",
        shuffle=True,
        batch_size=batch_size,
        num_workers=2,
    )
    val_data = load_data(
        "drive_data/val",
        transform_pipeline="default",
        shuffle=False,
        batch_size=batch_size,
        num_workers=2,
    )

    # Create loss functions
    seg_loss_func = torch.nn.CrossEntropyLoss()
    depth_loss_func = torch.nn.L1Loss()  # Mean Absolute Error for depth

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    global_step = 0
    best_iou = 0.0

    # Training loop
    for epoch in range(num_epoch):
        # Training phase
        model.train()
        train_metric = DetectionMetric(num_classes=3)
        train_metric.reset()

        for batch in train_data:
            # Move batch to device
            image = batch["image"].to(device)
            track = batch["track"].to(device)  # Segmentation labels
            depth = batch["depth"].to(device)  # Depth labels

            # Forward pass
            logits, pred_depth = model(image)

            # Compute losses
            seg_loss = seg_loss_func(logits, track)
            depth_loss = depth_loss_func(pred_depth, depth)

            # Combined loss
            total_loss = seg_weight * seg_loss + depth_weight * depth_loss

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Compute metrics
            pred = logits.argmax(dim=1)
            train_metric.add(pred, track, pred_depth, depth)

            # Log losses
            logger.add_scalar("train/seg_loss", seg_loss.item(), global_step)
            logger.add_scalar("train/depth_loss", depth_loss.item(), global_step)
            logger.add_scalar("train/total_loss", total_loss.item(), global_step)
            global_step += 1

        # Validation phase
        with torch.inference_mode():
            model.eval()
            val_metric = DetectionMetric(num_classes=3)
            val_metric.reset()

            for batch in val_data:
                # Move batch to device
                image = batch["image"].to(device)
                track = batch["track"].to(device)
                depth = batch["depth"].to(device)

                # Forward pass
                pred, pred_depth = model.predict(image)
                val_metric.add(pred, track, pred_depth, depth)

        # Compute metrics
        train_metrics = train_metric.compute()
        val_metrics = val_metric.compute()

        # Log metrics
        logger.add_scalar("train/iou", train_metrics["iou"], global_step)
        logger.add_scalar("train/accuracy", train_metrics["accuracy"], global_step)
        logger.add_scalar("train/depth_error", train_metrics["abs_depth_error"], global_step)
        logger.add_scalar("train/tp_depth_error", train_metrics["tp_depth_error"], global_step)

        logger.add_scalar("val/iou", val_metrics["iou"], global_step)
        logger.add_scalar("val/accuracy", val_metrics["accuracy"], global_step)
        logger.add_scalar("val/depth_error", val_metrics["abs_depth_error"], global_step)
        logger.add_scalar("val/tp_depth_error", val_metrics["tp_depth_error"], global_step)

        # Print progress
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:3d} / {num_epoch:3d}: "
                f"train_iou={train_metrics['iou']:.4f} "
                f"val_iou={val_metrics['iou']:.4f} "
                f"val_depth_err={val_metrics['abs_depth_error']:.4f}"
            )

        # Save best model based on IoU
        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            save_model(model)
            print(
                f"  -> New best model saved! "
                f"(iou={val_metrics['iou']:.4f}, "
                f"depth_err={val_metrics['abs_depth_error']:.4f})"
            )

    print(f"\nTraining complete! Best validation IoU: {best_iou:.4f}")
    print(f"Model saved to {save_model(model)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--seg_weight", type=float, default=1.0, help="Weight for segmentation loss")
    parser.add_argument("--depth_weight", type=float, default=1.0, help="Weight for depth loss")

    train(**vars(parser.parse_args()))

