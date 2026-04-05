"""Inference and evaluation"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from data.pets_dataset import OxfordIIITPetDataset
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet
from models.multitask import MultiTaskPerceptionModel


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
NUM_CLASSES = 37
NUM_SEGMENTS = 3
IN_CHANNELS = 3

DEFAULT_DATA_DIR = SCRIPT_DIR / "data"
DEFAULT_CHECKPOINT_DIR = SCRIPT_DIR / "checkpoints"
CHECKPOINT_NAMES = {
    "classification": "classifier.pth",
    "localization": "localizer.pth",
    "segmentation": "unet.pth",
}
MASK_COLOURS = np.array(
    [
        [20, 20, 20],
        [70, 160, 90],
        [230, 180, 60],
    ],
    dtype=np.uint8,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for evaluation."""
    parser = argparse.ArgumentParser(
        description="Run inference for Assignment-2 models."
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=("classification", "localization", "segmentation", "multitask"),
        help="Which model to evaluate.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Path to the Oxford-IIIT Pet data directory.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=("train", "val", "test"),
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Evaluation batch size."
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Square resize applied inside the dataset.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader workers.",
    )
    parser.add_argument(
        "--device",
        default=DEVICE,
        help="Torch device override. Defaults to cuda/mps/cpu auto-selection.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help="Directory containing the three mandatory checkpoint files.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Optional single-task checkpoint override.",
    )
    parser.add_argument(
        "--classifier-checkpoint",
        type=Path,
        default=None,
        help="Optional classifier checkpoint override for multitask inference.",
    )
    parser.add_argument(
        "--localizer-checkpoint",
        type=Path,
        default=None,
        help="Optional localizer checkpoint override for multitask inference.",
    )
    parser.add_argument(
        "--unet-checkpoint",
        type=Path,
        default=None,
        help="Optional segmentation checkpoint override for multitask inference.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional cap for quick smoke tests.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=SCRIPT_DIR / "outputs",
        help="Where qualitative visualisations should be written.",
    )
    parser.add_argument(
        "--num-visuals",
        type=int,
        default=5,
        help="Number of qualitative samples to save.",
    )
    return parser.parse_args()


def resolve_path_like(value: object, assignment_dir: Path) -> str:
    """Resolve relative manifest entries against the assignment root."""
    text = str(value)
    if text == "not_exist":
        return text
    path = Path(text)
    if path.is_absolute():
        return str(path)
    return str((assignment_dir / path).resolve())


def resolve_dataset_paths(dataset: OxfordIIITPetDataset, assignment_dir: Path) -> None:
    """Convert manifest rows to absolute paths once after loading the split."""
    dataset.df = dataset.df.copy()
    for column in ("image_path", "mask_path", "xml_path"):
        if column not in dataset.df.columns:
            continue
        dataset.df[column] = dataset.df[column].apply(
            lambda value: resolve_path_like(value, assignment_dir)
        )


def build_dataloader(
    data_dir: Path,
    split: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    device: str,
) -> DataLoader:
    """Create the evaluation dataloader."""
    dataset = OxfordIIITPetDataset(
        root_dir=str(data_dir), split=split, image_size=image_size
    )
    resolve_dataset_paths(dataset, SCRIPT_DIR)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device == "cuda",
    )


def load_checkpoint_state(
    checkpoint_path: Path, map_location: str = "cpu"
) -> Dict[str, torch.Tensor]:
    """Load a checkpoint that may be stored as a state_dict or wrapped payload."""
    payload = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(payload, dict) and "state_dict" in payload:
        return payload["state_dict"]
    if isinstance(payload, dict):
        return payload
    raise TypeError(f"Unsupported checkpoint format in {checkpoint_path}.")


def checkpoint_for_task(task: str, args: argparse.Namespace) -> Path:
    """Resolve the checkpoint path for one single-task model."""
    if args.checkpoint_path is not None:
        return args.checkpoint_path.expanduser().resolve()
    return (args.checkpoint_dir / CHECKPOINT_NAMES[task]).expanduser().resolve()


def build_model(args: argparse.Namespace):
    """Instantiate the requested model and load the required checkpoint(s)."""
    if args.task == "classification":
        checkpoint_path = checkpoint_for_task("classification", args)
        model = VGG11Classifier(
            num_classes=NUM_CLASSES,
            in_channels=IN_CHANNELS,
            dropout_p=0.5,
        )
        model.load_state_dict(
            load_checkpoint_state(checkpoint_path, map_location=args.device)
        )
        return model.to(args.device), {"classification": checkpoint_path}

    if args.task == "localization":
        checkpoint_path = checkpoint_for_task("localization", args)
        model = VGG11Localizer(in_channels=IN_CHANNELS, dropout_p=0.5)
        model.load_state_dict(
            load_checkpoint_state(checkpoint_path, map_location=args.device)
        )
        return model.to(args.device), {"localization": checkpoint_path}

    if args.task == "segmentation":
        checkpoint_path = checkpoint_for_task("segmentation", args)
        model = VGG11UNet(
            num_classes=NUM_SEGMENTS, in_channels=IN_CHANNELS, dropout_p=0.5
        )
        model.load_state_dict(
            load_checkpoint_state(checkpoint_path, map_location=args.device)
        )
        return model.to(args.device), {"segmentation": checkpoint_path}

    classifier_path = (
        args.classifier_checkpoint.expanduser().resolve()
        if args.classifier_checkpoint is not None
        else (args.checkpoint_dir / CHECKPOINT_NAMES["classification"])
        .expanduser()
        .resolve()
    )
    localizer_path = (
        args.localizer_checkpoint.expanduser().resolve()
        if args.localizer_checkpoint is not None
        else (args.checkpoint_dir / CHECKPOINT_NAMES["localization"])
        .expanduser()
        .resolve()
    )
    unet_path = (
        args.unet_checkpoint.expanduser().resolve()
        if args.unet_checkpoint is not None
        else (args.checkpoint_dir / CHECKPOINT_NAMES["segmentation"])
        .expanduser()
        .resolve()
    )

    model = MultiTaskPerceptionModel(
        num_breeds=NUM_CLASSES,
        seg_classes=NUM_SEGMENTS,
        in_channels=IN_CHANNELS,
        classifier_path=str(classifier_path),
        localizer_path=str(localizer_path),
        unet_path=str(unet_path),
    )
    return model.to(args.device), {
        "classification": classifier_path,
        "localization": localizer_path,
        "segmentation": unet_path,
    }


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
    """Compute a per-sample multi-class Dice score."""
    dice_scores = []
    for class_index in range(num_classes):
        pred_class = (predicted_mask == class_index).float()
        target_class = (target_mask == class_index).float()
        intersection = (pred_class * target_class).sum(dim=(1, 2))
        denominator = pred_class.sum(dim=(1, 2)) + target_class.sum(dim=(1, 2))
        dice_scores.append((2.0 * intersection + eps) / (denominator + eps))
    return torch.stack(dice_scores, dim=1).mean(dim=1)


def load_label_names(data_dir: Path) -> List[str]:
    """Derive readable breed labels from the annotation list file."""
    list_path = data_dir / "annotations" / "list.txt"
    names = {}
    if not list_path.exists():
        return [f"class_{index}" for index in range(NUM_CLASSES)]

    with list_path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            image_name, class_id, *_ = line.split()
            class_index = int(class_id) - 1
            breed_name = "_".join(image_name.split("_")[:-1]).replace("_", " ")
            names[class_index] = breed_name

    return [names.get(index, f"class_{index}") for index in range(NUM_CLASSES)]


def colourise_mask(mask: np.ndarray) -> np.ndarray:
    """Map class ids to an RGB image for saved visualisations."""
    mask = np.clip(mask.astype(np.int64), 0, len(MASK_COLOURS) - 1)
    return MASK_COLOURS[mask]


def tensor_image_to_numpy(image_tensor: torch.Tensor) -> np.ndarray:
    """Convert a CHW float tensor into an HWC numpy array."""
    image = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
    return np.clip(image, 0.0, 1.0)


def draw_boxes(
    ax, image: np.ndarray, gt_box: np.ndarray, pred_box: Optional[np.ndarray] = None
) -> None:
    """Draw ground-truth and prediction boxes onto one matplotlib axis."""
    ax.imshow(image)
    gt_box = cxcywh_to_xyxy(torch.tensor(gt_box).unsqueeze(0)).squeeze(0).numpy()
    gt_width = gt_box[2] - gt_box[0]
    gt_height = gt_box[3] - gt_box[1]
    gt_rect = plt.Rectangle(
        (gt_box[0], gt_box[1]),
        gt_width,
        gt_height,
        linewidth=2,
        edgecolor="lime",
        facecolor="none",
    )
    ax.add_patch(gt_rect)

    if pred_box is not None:
        pred_box = (
            cxcywh_to_xyxy(torch.tensor(pred_box).unsqueeze(0)).squeeze(0).numpy()
        )
        pred_width = pred_box[2] - pred_box[0]
        pred_height = pred_box[3] - pred_box[1]
        pred_rect = plt.Rectangle(
            (pred_box[0], pred_box[1]),
            pred_width,
            pred_height,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
        )
        ax.add_patch(pred_rect)

    ax.axis("off")


def save_visual(
    task: str,
    sample: Dict[str, np.ndarray],
    index: int,
    output_dir: Path,
    label_names: List[str],
) -> None:
    """Save one qualitative prediction figure."""
    output_dir.mkdir(parents=True, exist_ok=True)
    image = sample["image"]
    gt_label = int(sample["gt_label"])
    pred_label = int(sample["pred_label"])
    gt_box = sample["gt_box"]
    pred_box = sample["pred_box"]
    gt_mask = sample["gt_mask"]
    pred_mask = sample["pred_mask"]

    if task == "classification":
        fig, axis = plt.subplots(1, 1, figsize=(4, 4))
        axis.imshow(image)
        axis.set_title(f"GT: {label_names[gt_label]}\nPred: {label_names[pred_label]}")
        axis.axis("off")
    elif task == "localization":
        fig, axis = plt.subplots(1, 1, figsize=(5, 5))
        draw_boxes(axis, image, gt_box, pred_box)
        axis.set_title(f"IoU={sample['iou']:.3f}")
    else:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(image)
        axes[0].set_title(
            f"GT: {label_names[gt_label]}\nPred: {label_names[pred_label]}"
        )
        axes[0].axis("off")

        if task == "multitask":
            draw_boxes(axes[0], image, gt_box, pred_box)
            axes[0].set_title(
                f"GT: {label_names[gt_label]}\nPred: {label_names[pred_label]}\nIoU={sample['iou']:.3f}"
            )

        axes[1].imshow(colourise_mask(gt_mask))
        axes[1].set_title("Ground Truth Mask")
        axes[1].axis("off")

        axes[2].imshow(colourise_mask(pred_mask))
        axes[2].set_title("Predicted Mask")
        axes[2].axis("off")

    fig.tight_layout()
    fig.savefig(output_dir / f"{task}_{index:02d}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def evaluate(
    model,
    loader: DataLoader,
    task: str,
    device: str,
    num_visuals: int,
) -> tuple[Dict[str, float], List[Dict[str, np.ndarray]]]:
    """Run evaluation for a single-task or multitask model."""
    model.eval()

    total_examples = 0
    classification_targets = []
    classification_predictions = []
    localization_iou_sum = 0.0
    localization_hits50 = 0.0
    segmentation_correct_pixels = 0.0
    segmentation_total_pixels = 0.0
    segmentation_dice_sum = 0.0
    visuals = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device=device, dtype=torch.float32)
            gt_labels = batch["class_label"].to(device=device, dtype=torch.long)
            gt_boxes = batch["bbox"].to(device=device, dtype=torch.float32)
            gt_masks = batch["segmentation_mask"].to(device=device, dtype=torch.long)
            batch_size = images.shape[0]
            total_examples += batch_size

            if task == "classification":
                class_logits = model(images)
                class_predictions = class_logits.argmax(dim=1)
                classification_targets.extend(gt_labels.detach().cpu().tolist())
                classification_predictions.extend(
                    class_predictions.detach().cpu().tolist()
                )
            elif task == "localization":
                pred_boxes = model(images)
                ious = box_iou(pred_boxes, gt_boxes)
                localization_iou_sum += ious.sum().item()
                localization_hits50 += (ious >= 0.5).sum().item()
            elif task == "segmentation":
                seg_logits = model(images)
                pred_masks = seg_logits.argmax(dim=1)
                segmentation_correct_pixels += (pred_masks == gt_masks).sum().item()
                segmentation_total_pixels += gt_masks.numel()
                segmentation_dice_sum += (
                    batch_dice_score(pred_masks, gt_masks, num_classes=NUM_SEGMENTS)
                    .sum()
                    .item()
                )
            else:
                outputs = model(images)
                class_logits = outputs["classification"]
                pred_boxes = outputs["localization"]
                seg_logits = outputs["segmentation"]

                class_predictions = class_logits.argmax(dim=1)
                pred_masks = seg_logits.argmax(dim=1)
                ious = box_iou(pred_boxes, gt_boxes)

                classification_targets.extend(gt_labels.detach().cpu().tolist())
                classification_predictions.extend(
                    class_predictions.detach().cpu().tolist()
                )
                localization_iou_sum += ious.sum().item()
                localization_hits50 += (ious >= 0.5).sum().item()
                segmentation_correct_pixels += (pred_masks == gt_masks).sum().item()
                segmentation_total_pixels += gt_masks.numel()
                segmentation_dice_sum += (
                    batch_dice_score(pred_masks, gt_masks, num_classes=NUM_SEGMENTS)
                    .sum()
                    .item()
                )

            if len(visuals) < num_visuals:
                for item_index in range(batch_size):
                    if len(visuals) >= num_visuals:
                        break

                    sample = {
                        "image": tensor_image_to_numpy(images[item_index]),
                        "gt_label": int(gt_labels[item_index].item()),
                        "pred_label": int(gt_labels[item_index].item()),
                        "gt_box": gt_boxes[item_index].detach().cpu().numpy(),
                        "pred_box": gt_boxes[item_index].detach().cpu().numpy(),
                        "gt_mask": gt_masks[item_index].detach().cpu().numpy(),
                        "pred_mask": gt_masks[item_index].detach().cpu().numpy(),
                        "iou": 1.0,
                    }

                    if task == "classification":
                        sample["pred_label"] = int(class_predictions[item_index].item())
                    elif task == "localization":
                        sample["pred_box"] = (
                            pred_boxes[item_index].detach().cpu().numpy()
                        )
                        sample["iou"] = float(ious[item_index].item())
                    elif task == "segmentation":
                        sample["pred_mask"] = (
                            pred_masks[item_index].detach().cpu().numpy()
                        )
                    else:
                        sample["pred_label"] = int(class_predictions[item_index].item())
                        sample["pred_box"] = (
                            pred_boxes[item_index].detach().cpu().numpy()
                        )
                        sample["pred_mask"] = (
                            pred_masks[item_index].detach().cpu().numpy()
                        )
                        sample["iou"] = float(ious[item_index].item())

                    visuals.append(sample)

    metrics = {}

    if task in ("classification", "multitask"):
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

    if task in ("localization", "multitask"):
        metrics.update(
            {
                "mean_iou": localization_iou_sum / max(total_examples, 1),
                "ap50_proxy": localization_hits50 / max(total_examples, 1),
            }
        )

    if task in ("segmentation", "multitask"):
        metrics.update(
            {
                "pixel_accuracy": segmentation_correct_pixels
                / max(segmentation_total_pixels, 1),
                "dice": segmentation_dice_sum / max(total_examples, 1),
            }
        )

    return metrics, visuals


def format_metrics(metrics: Dict[str, float]) -> str:
    """Format metric values for a single terminal line."""
    return ", ".join(f"{key}={value:.4f}" for key, value in sorted(metrics.items()))


def main() -> None:
    """Run evaluation and save a few qualitative examples."""
    args = parse_args()
    args.data_dir = args.data_dir.expanduser().resolve()
    args.checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    args.save_dir = args.save_dir.expanduser().resolve()

    loader = build_dataloader(
        data_dir=args.data_dir,
        split=args.split,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )

    if args.max_batches is not None:
        total_items = args.max_batches * args.batch_size
        print(f"[setup] evaluating at most about {total_items} samples")

    model, checkpoint_paths = build_model(args)
    label_names = load_label_names(args.data_dir)

    if args.max_batches is not None:
        # Materialise a short iterator-backed list for smoke testing without changing DataLoader code.
        batches = []
        for batch_index, batch in enumerate(loader, start=1):
            batches.append(batch)
            if batch_index >= args.max_batches:
                break
        loader = batches  # type: ignore[assignment]

    metrics, visuals = evaluate(
        model=model,
        loader=loader,  # type: ignore[arg-type]
        task=args.task,
        device=args.device,
        num_visuals=args.num_visuals,
    )

    print(f"[checkpoints] {checkpoint_paths}")
    print(f"[metrics] {format_metrics(metrics)}")

    for sample_index, sample in enumerate(visuals):
        save_visual(
            task=args.task,
            sample=sample,
            index=sample_index,
            output_dir=args.save_dir,
            label_names=label_names,
        )

    print(f"[outputs] saved {len(visuals)} visual(s) to {args.save_dir}")


if __name__ == "__main__":
    main()
