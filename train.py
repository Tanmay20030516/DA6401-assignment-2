"""Training entrypoint"""

import argparse
import random
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from data.pets_dataset import OxfordIIITPetDataset
from losses.iou_loss import IoULoss
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet
from models.multitask import MultiTaskPerceptionModel

try:
    import wandb
except ImportError:
    wandb = None


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
NUM_CLASSES = 37
NUM_SEGMENTS = 3
IN_CHANNELS = 3

OPTIMIZER = optim.AdamW

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "data"
DEFAULT_CHECKPOINT_DIR = SCRIPT_DIR / "checkpoints"
CHECKPOINT_NAMES = {
    "classification": "classifier.pth",
    "localization": "localizer.pth",
    "segmentation": "unet.pth",
}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for single-task training."""
    parser = argparse.ArgumentParser(description="Train Assignment-2 models.")
    parser.add_argument(
        "--task",
        required=True,
        choices=sorted(CHECKPOINT_NAMES),
        help="Which single-task model to train.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Path to the Oxford-IIIT Pet data directory.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help="Where the mandatory checkpoint files should be written.",
    )
    parser.add_argument(
        "--encoder-checkpoint",
        type=Path,
        default=None,
        help="Optional classifier checkpoint used to warm-start the shared encoder.",
    )
    parser.add_argument(
        "--freeze-strategy",
        choices=("none", "encoder", "partial"),
        default="none",
        help="Freezing policy for localization / segmentation transfer experiments.",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay used by AdamW.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Square resize applied to images and masks.",
    )
    parser.add_argument(
        "--dropout-p",
        type=float,
        default=0.5,
        help="Dropout probability passed into the models.",
    )
    parser.add_argument(
        "--localization-loss",
        choices=("smoothl1", "l1", "mse", "iou"),
        default="smoothl1",
        help="Regression loss used for the localization head.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader workers. Keep this at 0 on MPS if you hit issues.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--device",
        default=DEVICE,
        help="Torch device override. Defaults to cuda/mps/cpu auto-selection.",
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Optional cap for smoke testing the training loop quickly.",
    )
    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=None,
        help="Optional cap for smoke testing the validation loop quickly.",
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Enable Weights & Biases scalar logging for this run.",
    )
    parser.add_argument(
        "--no-bn",
        action="store_true",
        help="Disable BatchNorm in the encoder (for ablation study).",
    )
    parser.add_argument(
        "--wandb-project",
        default="da6401_assignment2",
        help="W&B project name used when --use-wandb is enabled.",
    )
    parser.add_argument(
        "--wandb-entity",
        default=None,
        help="Optional W&B entity / team.",
    )
    parser.add_argument(
        "--group",
        type=str,
        default=None,
        required=False,
        help="Group name for W&B project",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional explicit run name for W&B.",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    """Set seeds so short smoke tests are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_dataset_paths(dataset: OxfordIIITPetDataset, assignment_dir: Path) -> None:
    """Manifest rows use assignment-root relative paths, so convert them once."""
    dataset.df = dataset.df.copy()
    for column in ("image_path", "mask_path", "xml_path"):
        if column not in dataset.df.columns:
            continue
        dataset.df[column] = dataset.df[column].apply(
            lambda value: resolve_path_like(value, assignment_dir)
        )


def resolve_path_like(value: object, assignment_dir: Path) -> str:
    """Resolve relative manifest paths without touching the sentinel `not_exist` value."""
    text = str(value)
    if text == "not_exist":
        return text
    path = Path(text)
    if path.is_absolute():
        return str(path)
    return str((assignment_dir / path).resolve())


def build_dataloader(
    data_dir: Path,
    split: str,
    image_size: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: str,
) -> DataLoader:
    """Create one dataloader for a requested split."""
    dataset = OxfordIIITPetDataset(root_dir=str(data_dir), split=split, image_size=image_size)
    resolve_dataset_paths(dataset, SCRIPT_DIR)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device == "cuda",
    )


def load_checkpoint_state(checkpoint_path: Path, map_location: str = "cpu") -> Dict[str, torch.Tensor]:
    """Load a checkpoint that may be either a bare state_dict or a wrapped payload."""
    payload = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(payload, dict) and "state_dict" in payload:
        return payload["state_dict"]
    if isinstance(payload, dict):
        return payload
    raise TypeError(f"Unsupported checkpoint format in {checkpoint_path}.")


def load_encoder_weights(model: nn.Module, checkpoint_path: Path) -> None:
    """Warm-start the encoder from a trained classifier checkpoint."""
    state_dict = load_checkpoint_state(checkpoint_path, map_location="cpu")
    encoder_state = {}
    for key, value in state_dict.items():
        if key.startswith("encoder."):
            encoder_state[key.removeprefix("encoder.")] = value

    if not encoder_state:
        raise ValueError(
            f"No encoder weights were found inside {checkpoint_path}. "
            "Expected keys starting with `encoder.`."
        )

    missing, unexpected = model.encoder.load_state_dict(encoder_state, strict=False)
    if missing:
        print(f"[encoder-load] Missing keys: {missing}")
    if unexpected:
        print(f"[encoder-load] Unexpected keys: {unexpected}")


def configure_freeze_strategy(model: nn.Module, strategy: str) -> None:
    """Apply the requested transfer-learning policy to encoder-based models."""
    if not hasattr(model, "encoder"):
        return

    for parameter in model.encoder.parameters():
        parameter.requires_grad = True

    if strategy == "encoder":
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False
        return

    if strategy == "partial":
        # Freeze earlier blocks and leave the deeper blocks trainable.
        frozen_prefixes = (
            "conv1",
            "bn1",
            "conv2",
            "bn2",
            "conv3",
            "bn3",
            "conv4",
            "bn4",
        )
        for name, parameter in model.encoder.named_parameters():
            if name.startswith(frozen_prefixes):
                parameter.requires_grad = False


def build_model(args: argparse.Namespace) -> nn.Module:
    """Instantiate the requested model and apply optional transfer settings."""
    if args.task == "classification":
        model = VGG11Classifier(
            num_classes=NUM_CLASSES,
            in_channels=IN_CHANNELS,
            dropout_p=args.dropout_p,
            use_bn=not args.no_bn,
        )
    elif args.task == "localization":
        model = VGG11Localizer(in_channels=IN_CHANNELS, dropout_p=args.dropout_p)
    else:
        model = VGG11UNet(
            num_classes=NUM_SEGMENTS,
            in_channels=IN_CHANNELS,
            dropout_p=args.dropout_p,
        )

    if args.encoder_checkpoint is not None and args.task != "classification":
        load_encoder_weights(model, args.encoder_checkpoint)

    if args.task != "classification":
        configure_freeze_strategy(model, args.freeze_strategy)

    return model


def build_criterion(args: argparse.Namespace) -> nn.Module:
    """Select a loss that matches the chosen task."""
    if args.task == "classification":
        return nn.CrossEntropyLoss()
    if args.task == "segmentation":
        return nn.CrossEntropyLoss()
    if args.localization_loss == "l1":
        return nn.L1Loss()
    if args.localization_loss == "mse":
        return nn.MSELoss()
    if args.localization_loss == "iou":
        return IoULoss()
    return nn.SmoothL1Loss(beta=1.0)


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Count parameters for a quick sanity check before training."""
    parameters = model.parameters()
    if trainable_only:
        parameters = (parameter for parameter in parameters if parameter.requires_grad)
    return sum(parameter.numel() for parameter in parameters)


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert centre-width-height boxes into corner coordinates."""
    x_center, y_center, width, height = boxes.unbind(dim=1)
    half_width = width / 2.0
    half_height = height / 2.0
    return torch.stack(
        (
            x_center - half_width,
            y_center - half_height,
            x_center + half_width,
            y_center + half_height,
        ),
        dim=1,
    )


def box_iou(pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
    """Compute IoU for aligned batches of predicted and target boxes."""
    pred_xyxy = cxcywh_to_xyxy(pred_boxes)
    target_xyxy = cxcywh_to_xyxy(target_boxes)

    top_left = torch.maximum(pred_xyxy[:, :2], target_xyxy[:, :2])
    bottom_right = torch.minimum(pred_xyxy[:, 2:], target_xyxy[:, 2:])
    wh = (bottom_right - top_left).clamp(min=0.0)
    intersection = wh[:, 0] * wh[:, 1]

    pred_area = (pred_xyxy[:, 2] - pred_xyxy[:, 0]).clamp(min=0.0) * (
        pred_xyxy[:, 3] - pred_xyxy[:, 1]
    ).clamp(min=0.0)
    target_area = (target_xyxy[:, 2] - target_xyxy[:, 0]).clamp(min=0.0) * (
        target_xyxy[:, 3] - target_xyxy[:, 1]
    ).clamp(min=0.0)
    union = pred_area + target_area - intersection

    return intersection / union.clamp(min=1e-6)


def batch_dice_score(
    predicted_mask: torch.Tensor,
    target_mask: torch.Tensor,
    num_classes: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute a mean multi-class Dice score for each sample in the batch."""
    dice_scores = []
    for class_index in range(num_classes):
        pred_class = (predicted_mask == class_index).float()
        target_class = (target_mask == class_index).float()
        intersection = (pred_class * target_class).sum(dim=(1, 2))
        denominator = pred_class.sum(dim=(1, 2)) + target_class.sum(dim=(1, 2))
        dice_scores.append((2.0 * intersection + eps) / (denominator + eps))
    return torch.stack(dice_scores, dim=1).mean(dim=1)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    task: str,
    device: str,
    optimizer: Optional[optim.Optimizer] = None,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """Run one training or validation epoch and return aggregate metrics."""
    is_training = optimizer is not None
    if is_training:
        model.train()
    else:
        model.eval()

    total_examples = 0
    total_loss = 0.0

    classification_targets = []
    classification_predictions = []

    localization_iou_sum = 0.0
    localization_l1_sum = 0.0
    localization_hits50 = 0.0

    segmentation_correct_pixels = 0.0
    segmentation_total_pixels = 0.0
    segmentation_dice_sum = 0.0

    for batch_index, batch in enumerate(loader, start=1):
        images = batch["image"].to(device=device, dtype=torch.float32)
        batch_size = images.shape[0]
        total_examples += batch_size

        with torch.set_grad_enabled(is_training):
            outputs = model(images)

            if task == "classification":
                targets = batch["class_label"].to(device=device, dtype=torch.long)
                loss = criterion(outputs, targets)
                predictions = outputs.argmax(dim=1)
                classification_targets.extend(targets.detach().cpu().tolist())
                classification_predictions.extend(predictions.detach().cpu().tolist())

            elif task == "localization":
                targets = batch["bbox"].to(device=device, dtype=torch.float32)
                loss = criterion(outputs, targets)
                ious = box_iou(outputs.detach(), targets)
                localization_iou_sum += ious.sum().item()
                localization_l1_sum += torch.abs(outputs.detach() - targets).sum().item()
                localization_hits50 += (ious >= 0.5).sum().item()

            else:
                targets = batch["segmentation_mask"].to(device=device, dtype=torch.long)
                loss = criterion(outputs, targets)
                predictions = outputs.argmax(dim=1)
                segmentation_correct_pixels += (predictions == targets).sum().item()
                segmentation_total_pixels += targets.numel()
                segmentation_dice_sum += batch_dice_score(
                    predictions, targets, num_classes=NUM_SEGMENTS
                ).sum().item()

            if is_training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * batch_size

        if max_batches is not None and batch_index >= max_batches:
            break

    metrics = {
        "loss": total_loss / max(total_examples, 1),
    }

    if task == "classification":
        accuracy = (
            np.mean(np.equal(classification_targets, classification_predictions))
            if classification_targets
            else 0.0
        )
        macro_f1 = (
            f1_score(
                classification_targets,
                classification_predictions,
                average="macro",
                zero_division=0,
            )
            if classification_targets
            else 0.0
        )
        metrics.update({"accuracy": float(accuracy), "macro_f1": float(macro_f1)})

    elif task == "localization":
        metrics.update(
            {
                "mean_iou": localization_iou_sum / max(total_examples, 1),
                "l1_error": localization_l1_sum / max(total_examples * 4, 1),
                "ap50_proxy": localization_hits50 / max(total_examples, 1),
            }
        )

    else:
        metrics.update(
            {
                "pixel_accuracy": segmentation_correct_pixels
                / max(segmentation_total_pixels, 1),
                "dice": segmentation_dice_sum / max(total_examples, 1),
            }
        )

    return metrics


def select_model_metric(task: str) -> str:
    """Pick the validation metric that decides the best checkpoint."""
    if task == "classification":
        return "macro_f1"
    if task == "localization":
        return "mean_iou"
    return "dice"


def format_metrics(metrics: Dict[str, float]) -> str:
    """Pretty-print a metrics dictionary with stable ordering."""
    return ", ".join(f"{key}={value:.4f}" for key, value in sorted(metrics.items()))


def checkpoint_path_for_task(task: str, checkpoint_dir: Path) -> Path:
    """Return the mandatory grading filename for the chosen task."""
    return checkpoint_dir / CHECKPOINT_NAMES[task]


def save_checkpoint(model: nn.Module, checkpoint_path: Path) -> None:
    """Save a CPU state_dict so the provided multi-task wrapper can load it anywhere."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    cpu_state = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
    }
    torch.save(cpu_state, checkpoint_path)


def maybe_init_wandb(model: nn.Module, args: argparse.Namespace):
    """Start a W&B run only when explicitly requested by the caller."""
    if not args.use_wandb:
        return None

    if wandb is None:
        raise ImportError(
            "wandb is not installed in the active environment. "
            "Install it or train without --use-wandb."
        )

    run_name = args.run_name or f"{args.task}-{args.freeze_strategy}"
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.group,
        name=run_name,
        config={
            "task": args.task,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "dropout_p": args.dropout_p,
            "freeze_strategy": args.freeze_strategy,
            "image_size": args.image_size,
            "localization_loss": args.localization_loss,
        },
    )
    run.watch(model, log="gradients", log_freq=100)
    return run


def validate_multitask_bundle(checkpoint_dir: Path) -> None:
    """Smoke-test the unified pipeline once all three mandatory checkpoints exist."""
    classifier_path = checkpoint_dir / CHECKPOINT_NAMES["classification"]
    localizer_path = checkpoint_dir / CHECKPOINT_NAMES["localization"]
    unet_path = checkpoint_dir / CHECKPOINT_NAMES["segmentation"]

    if not (classifier_path.exists() and localizer_path.exists() and unet_path.exists()):
        return

    try:
        bundle = MultiTaskPerceptionModel(
            num_breeds=NUM_CLASSES,
            seg_classes=NUM_SEGMENTS,
            in_channels=IN_CHANNELS,
            classifier_path=str(classifier_path),
            localizer_path=str(localizer_path),
            unet_path=str(unet_path),
        )
        del bundle
        print("[multitask-check] All three checkpoints can be loaded by MultiTaskPerceptionModel.")
    except Exception as error:
        print(f"[multitask-check] Failed to build unified model: {error}")


def main() -> None:
    """Run the end-to-end single-task training loop."""
    args = parse_args()
    args.data_dir = args.data_dir.expanduser().resolve()
    args.checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    if args.encoder_checkpoint is not None:
        args.encoder_checkpoint = args.encoder_checkpoint.expanduser().resolve()

    seed_everything(args.seed)

    print(f"[setup] device={args.device}")
    print(f"[setup] data_dir={args.data_dir}")
    print(f"[setup] checkpoint_dir={args.checkpoint_dir}")

    train_loader = build_dataloader(
        data_dir=args.data_dir,
        split="train",
        image_size=args.image_size,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        device=args.device,
    )
    val_loader = build_dataloader(
        data_dir=args.data_dir,
        split="val",
        image_size=args.image_size,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        device=args.device,
    )

    model = build_model(args).to(args.device)
    criterion = build_criterion(args)

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise RuntimeError("No trainable parameters remain after applying the freeze strategy.")

    optimizer = OPTIMIZER(
        trainable_parameters,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    print(f"[model] total_params={count_parameters(model):,}")
    print(f"[model] trainable_params={count_parameters(model, trainable_only=True):,}")

    run = maybe_init_wandb(model, args)

    best_metric_name = select_model_metric(args.task)
    best_metric_value = float("-inf")
    checkpoint_path = checkpoint_path_for_task(args.task, args.checkpoint_dir)

    ## for task 2.1
    fixed_images = next(iter(val_loader))["image"].to(args.device, dtype=torch.float32)

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            task=args.task,
            device=args.device,
            optimizer=optimizer,
            max_batches=args.max_train_batches,
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            task=args.task,
            device=args.device,
            optimizer=None,
            max_batches=args.max_val_batches,
        )
        ## for 2.1 task
        if run is not None and args.task == "classification":
            activations = {}
            # we look at conv3 layer outputs after batchnorm
            # relu is a shared non-linearity, so we can't capture it
            # but relu is just clipping between 0 and inf, so we can still get a sense of activation magnitudes without it
            handle = model.encoder.bn3.register_forward_hook(
                lambda m, inp, out: activations.update({"conv3": out.detach().cpu()})
            )
            model.eval()
            with torch.no_grad():
                model(fixed_images)
            handle.remove()
            model.train() # restore training mode
            run.log({
                "epoch": epoch,
                "conv3_activations": wandb.Histogram(activations["conv3"].numpy().flatten()),
            })

        current_metric = val_metrics[best_metric_name]
        improved = current_metric > best_metric_value
        if improved:
            best_metric_value = current_metric
            save_checkpoint(model, checkpoint_path)

        epoch_time = time.time() - epoch_start
        status = "saved" if improved else "kept"
        print(
            f"[epoch {epoch:03d}] "
            f"train: {format_metrics(train_metrics)} | "
            f"val: {format_metrics(val_metrics)} | "
            f"time={epoch_time:.1f}s | checkpoint={status}"
        )
        scheduler.step()
        if run is not None:
            run.log(
                {
                    "epoch": epoch,
                    "lr": scheduler.get_last_lr()[0], #optimizer.param_groups[0]["lr"],
                    **{f"train/{key}": value for key, value in train_metrics.items()},
                    **{f"val/{key}": value for key, value in val_metrics.items()},
                }
            )

    print(f"[done] best_val_{best_metric_name}={best_metric_value:.4f}")
    print(f"[done] best checkpoint saved to {checkpoint_path}")

    validate_multitask_bundle(args.checkpoint_dir)

    if run is not None:
        run.summary[f"best_val_{best_metric_name}"] = best_metric_value
        run.summary["checkpoint_path"] = str(checkpoint_path)
        run.finish()


if __name__ == "__main__":
    main()
