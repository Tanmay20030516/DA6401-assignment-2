import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional

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

SCRIPT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".matplotlib"))

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
        [70, 160, 90],  # 0 = body (green)
        [20, 20, 20],  # 1 = background (dark)
        [230, 180, 60],
    ],  # 2 = boundary (orange)
    dtype=np.uint8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference for Assignment-2 models."
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=("classification", "localization", "segmentation", "multitask"),
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="single-task checkpoint override.",
    )
    parser.add_argument("--classifier-checkpoint", type=Path, default=None)
    parser.add_argument("--localizer-checkpoint", type=Path, default=None)
    parser.add_argument("--unet-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="cap number of batches for quick smoke tests.",
    )
    parser.add_argument("--save-dir", type=Path, default=SCRIPT_DIR / "outputs")
    parser.add_argument("--num-visuals", type=int, default=5)
    return parser.parse_args()


def load_checkpoint_state(path: Path, map_location="cpu") -> Dict[str, torch.Tensor]:
    payload = torch.load(path, map_location=map_location)
    if isinstance(payload, dict) and "state_dict" in payload:
        return payload["state_dict"]
    if isinstance(payload, dict):
        return payload
    raise TypeError(f"unsupported checkpoint format in {path}.")


def _resolve_path(override: Optional[Path], default: Path) -> Path:
    return (
        override.expanduser().resolve()
        if override is not None
        else default.expanduser().resolve()
    )


def _load_single(model: torch.nn.Module, path: Path, device: str) -> torch.nn.Module:
    model.load_state_dict(load_checkpoint_state(path, map_location=device))
    return model.to(device)


def build_model(args: argparse.Namespace):
    ckpt_dir = args.checkpoint_dir

    if args.task == "classification":
        path = _resolve_path(
            args.checkpoint_path, ckpt_dir / CHECKPOINT_NAMES["classification"]
        )
        model = _load_single(
            VGG11Classifier(NUM_CLASSES, IN_CHANNELS, dropout_p=0.5), path, args.device
        )
        return model, {"classification": path}

    if args.task == "localization":
        path = _resolve_path(
            args.checkpoint_path, ckpt_dir / CHECKPOINT_NAMES["localization"]
        )
        model = _load_single(
            VGG11Localizer(IN_CHANNELS, dropout_p=0.5, image_size=args.image_size),
            path,
            args.device,
        )
        return model, {"localization": path}

    if args.task == "segmentation":
        path = _resolve_path(
            args.checkpoint_path, ckpt_dir / CHECKPOINT_NAMES["segmentation"]
        )
        model = _load_single(
            VGG11UNet(NUM_SEGMENTS, IN_CHANNELS, dropout_p=0.5), path, args.device
        )
        return model, {"segmentation": path}

    # for multitask, resolve each checkpoint independently
    clf_path = _resolve_path(
        args.classifier_checkpoint, ckpt_dir / CHECKPOINT_NAMES["classification"]
    )
    loc_path = _resolve_path(
        args.localizer_checkpoint, ckpt_dir / CHECKPOINT_NAMES["localization"]
    )
    seg_path = _resolve_path(
        args.unet_checkpoint, ckpt_dir / CHECKPOINT_NAMES["segmentation"]
    )
    model = MultiTaskPerceptionModel(
        num_breeds=NUM_CLASSES,
        seg_classes=NUM_SEGMENTS,
        in_channels=IN_CHANNELS,
        classifier_path=str(clf_path),
        localizer_path=str(loc_path),
        unet_path=str(seg_path),
    )
    return model.to(args.device), {
        "classification": clf_path,
        "localization": loc_path,
        "segmentation": seg_path,
    }


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(dim=1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1)


def box_iou(pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
    p = cxcywh_to_xyxy(pred_boxes)
    t = cxcywh_to_xyxy(target_boxes)
    tl = torch.maximum(p[:, :2], t[:, :2])
    br = torch.minimum(p[:, 2:], t[:, 2:])
    wh = (br - tl).clamp(min=0.0)
    inter = wh[:, 0] * wh[:, 1]
    pred_area = (p[:, 2] - p[:, 0]).clamp(0) * (p[:, 3] - p[:, 1]).clamp(0)
    tgt_area = (t[:, 2] - t[:, 0]).clamp(0) * (t[:, 3] - t[:, 1]).clamp(0)
    return inter / (pred_area + tgt_area - inter).clamp(min=1e-6)


def batch_dice_score(pred_mask, target_mask, num_classes, eps=1e-6) -> torch.Tensor:
    scores = []
    for c in range(num_classes):
        p = (pred_mask == c).float()
        t = (target_mask == c).float()
        inter = (p * t).sum(dim=(1, 2))
        denom = p.sum(dim=(1, 2)) + t.sum(dim=(1, 2))
        scores.append((2.0 * inter + eps) / (denom + eps))
    return torch.stack(scores, dim=1).mean(dim=1)


def load_label_names(data_dir: Path) -> List[str]:
    list_path = data_dir / "annotations" / "list.txt"
    if not list_path.exists():
        return [f"class_{i}" for i in range(NUM_CLASSES)]
    names = {}
    with list_path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            image_name, class_id, *_ = line.split()
            names[int(class_id) - 1] = "_".join(image_name.split("_")[:-1]).replace(
                "_", " "
            )
    return [names.get(i, f"class_{i}") for i in range(NUM_CLASSES)]


def colourise_mask(mask: np.ndarray) -> np.ndarray:
    return MASK_COLOURS[np.clip(mask.astype(np.int64), 0, len(MASK_COLOURS) - 1)]


def to_numpy(img_tensor: torch.Tensor) -> np.ndarray:
    return np.clip(img_tensor.detach().cpu().permute(1, 2, 0).numpy(), 0.0, 1.0)


def draw_boxes(
    ax, image: np.ndarray, gt_box: np.ndarray, pred_box: Optional[np.ndarray] = None
) -> None:
    ax.imshow(image)

    # convert cxcywh numpy arrays -> xyxy for matplotlib Rectangle
    def _to_rect(box_np, color):
        b = cxcywh_to_xyxy(torch.tensor(box_np).unsqueeze(0)).squeeze(0).numpy()
        return plt.Rectangle(
            (b[0], b[1]),
            b[2] - b[0],
            b[3] - b[1],
            linewidth=2,
            edgecolor=color,
            facecolor="none",
        )

    ax.add_patch(_to_rect(gt_box, "lime"))
    if pred_box is not None:
        ax.add_patch(_to_rect(pred_box, "red"))
    ax.axis("off")


def save_visual(
    task: str, sample: Dict, index: int, output_dir: Path, label_names: List[str]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    image = sample["image"]
    gt_lbl, pred_lbl = (
        label_names[sample["gt_label"]],
        label_names[sample["pred_label"]],
    )

    if task == "classification":
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.imshow(image)
        ax.set_title(f"GT: {gt_lbl}\nPred: {pred_lbl}")
        ax.axis("off")

    elif task == "localization":
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        draw_boxes(ax, image, sample["gt_box"], sample["pred_box"])
        ax.set_title(f"IoU={sample['iou']:.3f}")

    else:  # segmentation or multitask
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        if task == "multitask":
            draw_boxes(axes[0], image, sample["gt_box"], sample["pred_box"])
            axes[0].set_title(
                f"GT: {gt_lbl}\nPred: {pred_lbl}\nIoU={sample['iou']:.3f}"
            )
        else:
            axes[0].imshow(image)
            axes[0].set_title(f"GT: {gt_lbl}\nPred: {pred_lbl}")
            axes[0].axis("off")
        axes[1].imshow(colourise_mask(sample["gt_mask"]))
        axes[1].set_title("Ground Truth Mask")
        axes[1].axis("off")
        axes[2].imshow(colourise_mask(sample["pred_mask"]))
        axes[2].set_title("Predicted Mask")
        axes[2].axis("off")

    fig.tight_layout()
    fig.savefig(output_dir / f"{task}_{index:02d}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def evaluate(model, loader, task: str, device: str, num_visuals: int):
    model.eval()

    total = 0
    cls_targets, cls_preds = [], []
    loc_iou_sum = loc_hits50 = 0.0
    seg_correct = seg_total = seg_dice_sum = 0.0
    visuals = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device=device, dtype=torch.float32)
            gt_labels = batch["class_label"].to(device=device, dtype=torch.long)
            gt_boxes = batch["bbox"].to(device=device, dtype=torch.float32)
            gt_masks = batch["segmentation_mask"].to(device=device, dtype=torch.long)
            bs = images.shape[0]
            total += bs

            if task == "classification":
                logits = model(images)
                preds = logits.argmax(dim=1)
                cls_targets.extend(gt_labels.cpu().tolist())
                cls_preds.extend(preds.cpu().tolist())

            elif task == "localization":
                pred_boxes = model(images)
                ious = box_iou(pred_boxes, gt_boxes)
                loc_iou_sum += ious.sum().item()
                loc_hits50 += (ious >= 0.5).sum().item()

            elif task == "segmentation":
                pred_masks = model(images).argmax(dim=1)
                seg_correct += (pred_masks == gt_masks).sum().item()
                seg_total += gt_masks.numel()
                seg_dice_sum += (
                    batch_dice_score(pred_masks, gt_masks, NUM_SEGMENTS).sum().item()
                )

            else:  # multitask
                out = model(images)
                logits = out["classification"]
                pred_boxes = out["localization"]
                pred_masks = out["segmentation"].argmax(dim=1)
                preds = logits.argmax(dim=1)
                ious = box_iou(pred_boxes, gt_boxes)
                cls_targets.extend(gt_labels.cpu().tolist())
                cls_preds.extend(preds.cpu().tolist())
                loc_iou_sum += ious.sum().item()
                loc_hits50 += (ious >= 0.5).sum().item()
                seg_correct += (pred_masks == gt_masks).sum().item()
                seg_total += gt_masks.numel()
                seg_dice_sum += (
                    batch_dice_score(pred_masks, gt_masks, NUM_SEGMENTS).sum().item()
                )

            # collect visuals — default everything to gt so unused fields are valid
            for i in range(bs):
                if len(visuals) >= num_visuals:
                    break
                sample = {
                    "image": to_numpy(images[i]),
                    "gt_label": int(gt_labels[i].item()),
                    "pred_label": int(
                        gt_labels[i].item()
                    ),  # overwritten below if needed
                    "gt_box": gt_boxes[i].cpu().numpy(),
                    "pred_box": gt_boxes[i].cpu().numpy(),
                    "gt_mask": gt_masks[i].cpu().numpy(),
                    "pred_mask": gt_masks[i].cpu().numpy(),
                    "iou": 1.0,
                }
                if task == "classification":
                    sample["pred_label"] = int(preds[i].item())
                elif task == "localization":
                    sample["pred_box"] = pred_boxes[i].cpu().numpy()
                    sample["iou"] = float(ious[i].item())
                elif task == "segmentation":
                    sample["pred_mask"] = pred_masks[i].cpu().numpy()
                else:
                    sample["pred_label"] = int(preds[i].item())
                    sample["pred_box"] = pred_boxes[i].cpu().numpy()
                    sample["pred_mask"] = pred_masks[i].cpu().numpy()
                    sample["iou"] = float(ious[i].item())
                visuals.append(sample)

    metrics = {}
    if task in ("classification", "multitask"):
        metrics["accuracy"] = (
            float(np.mean(np.equal(cls_targets, cls_preds))) if cls_targets else 0.0
        )
        metrics["macro_f1"] = (
            float(f1_score(cls_targets, cls_preds, average="macro", zero_division=0))
            if cls_targets
            else 0.0
        )
    if task in ("localization", "multitask"):
        metrics["mean_iou"] = loc_iou_sum / max(total, 1)
        metrics["ap50_proxy"] = loc_hits50 / max(total, 1)
    if task in ("segmentation", "multitask"):
        metrics["pixel_accuracy"] = seg_correct / max(seg_total, 1)
        metrics["dice"] = seg_dice_sum / max(total, 1)

    return metrics, visuals


def main() -> None:
    args = parse_args()
    args.data_dir = args.data_dir.expanduser().resolve()
    args.checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    args.save_dir = args.save_dir.expanduser().resolve()

    dataset = OxfordIIITPetDataset(
        root_dir=str(args.data_dir), split=args.split, image_size=args.image_size
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device == "cuda",
    )

    model, checkpoint_paths = build_model(args)
    label_names = load_label_names(args.data_dir)

    metrics, visuals = evaluate(model, loader, args.task, args.device, args.num_visuals)  # type: ignore

    print(f"[checkpoints] {checkpoint_paths}")
    print(f"[metrics] {', '.join(f'{k}={v:.4f}' for k, v in sorted(metrics.items()))}")

    for idx, sample in enumerate(visuals):
        save_visual(args.task, sample, idx, args.save_dir, label_names)
    print(f"[outputs] saved {len(visuals)} visual(s) to {args.save_dir}")


if __name__ == "__main__":
    main()
