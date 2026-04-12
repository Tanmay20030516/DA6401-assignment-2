import argparse
import random
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from data.pets_dataset import OxfordIIITPetDataset
from losses.iou_loss import IoULoss
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet

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

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "data"
DEFAULT_CHECKPOINT_DIR = SCRIPT_DIR / "checkpoints"
CHECKPOINT_NAMES = {
    "classification": "classifier.pth",
    "localization": "localizer.pth",
    "segmentation": "unet.pth",
}
# metric to monitor for checkpoint saving as per task
BEST_METRIC = {"classification": "macro_f1", "localization": "mean_iou", "segmentation": "dice"}

# some custom losses to be used for tasks
class SegmentationLoss(torch.nn.Module):
    # combined weighted CE + soft dice; pure CE struggles with the boundary class imbalance
    def __init__(self, num_classes=3, eps=1e-6):
        super().__init__()
        # body=0, background=1, boundary=2 (upweight the minority boundary class)
        self.ce = torch.nn.CrossEntropyLoss(weight=torch.tensor([3.0, 1.0, 4.0], device=DEVICE))
        self.eps = eps
        self.num_classes = num_classes
 
    def dice_loss(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        total = 0.0
        for c in range(self.num_classes):
            p = probs[:, c]
            t = (targets == c).float()
            intersection = (p * t).sum()
            denom = p.sum() + t.sum()
            total += 1.0 - (2.0 * intersection + self.eps) / (denom + self.eps)
        return total / self.num_classes
 
    def forward(self, logits, targets):
        return 1.0 * self.ce(logits, targets) + 2.0 * self.dice_loss(logits, targets)
 
 
class CombinedLocalizationLoss(torch.nn.Module):
    # iou loss alone kills gradients at init (negative w/h -> zero grad), so we pair it with mse/smooth l1
    def __init__(self, image_size=224, alpha=2.0, beta=1.0, use_l1=True):
        super().__init__()
        self.iou = IoULoss()
        # self.mse = torch.nn.MSELoss() # not that good of loss
        self.smooth_l1 = torch.nn.SmoothL1Loss()
        self.image_size = image_size
        self.alpha = alpha # weight for iou loss
        self.beta = beta # weight for mse loss / smoothl1 loss
 
    def forward(self, pred, target):
        iou_loss = self.iou(pred, target)
        # normalize coords to [0, 1] so mse magnitude is comparable to iou loss
        pred_n = pred / self.image_size
        target_n = target / self.image_size
        return self.alpha * iou_loss + self.beta * self.smooth_l1(pred_n, target_n)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Assignment-2 models.")
    parser.add_argument("--task", required=True, choices=sorted(CHECKPOINT_NAMES))
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--encoder-checkpoint", type=Path, default=None,
                        help="classifier checkpoint to warm-start the encoder.")
    parser.add_argument("--freeze-strategy", choices=("none", "encoder", "partial"), default="none")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--dropout-p", type=float, default=0.5)
    parser.add_argument("--localization-loss", choices=("smoothl1", "l1", "mse", "iou"), default="smoothl1")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--no-bn", action="store_true", help="disable batchnorm (ablation)")
    parser.add_argument("--wandb-project", default="da6401_assignment2")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--group", type=str, default=None)
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataloader(data_dir, split, image_size, batch_size, shuffle, num_workers, device) -> DataLoader:
    dataset = OxfordIIITPetDataset(root_dir=str(data_dir), split=split, image_size=image_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=device == "cuda")


def load_checkpoint_state(checkpoint_path: Path, map_location="cpu") -> Dict[str, torch.Tensor]:
    payload = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(payload, dict) and "state_dict" in payload:
        return payload["state_dict"]
    if isinstance(payload, dict):
        return payload
    raise TypeError(f"Unsupported checkpoint format in {checkpoint_path}.")


def load_encoder_weights(model: torch.nn.Module, checkpoint_path: Path) -> None:
    state_dict = load_checkpoint_state(checkpoint_path, map_location="cpu")
    encoder_state = {
        k.removeprefix("encoder."): v
        for k, v in state_dict.items() if k.startswith("encoder.")
    }
    if not encoder_state:
        raise ValueError(f"No encoder.* keys found in {checkpoint_path}.")
    missing, unexpected = model.encoder.load_state_dict(encoder_state, strict=False)
    if missing:
        print(f"[encoder-load] missing keys: {missing}")
    if unexpected:
        print(f"[encoder-load] unexpected keys: {unexpected}")


def configure_freeze_strategy(model: torch.nn.Module, strategy: str) -> None:
    if not hasattr(model, "encoder"):
        return
    for p in model.encoder.parameters():
        p.requires_grad = True
    if strategy == "encoder":
        for p in model.encoder.parameters():
            p.requires_grad = False
    elif strategy == "partial":
        # freeze early blocks; keep deep blocks trainable
        frozen_prefixes = ("conv1", "bn1", "conv2", "bn2", "conv3", "bn3", "conv4", "bn4")
        for name, p in model.encoder.named_parameters():
            if name.startswith(frozen_prefixes):
                p.requires_grad = False


def build_model(args: argparse.Namespace) -> torch.nn.Module:
    if args.task == "classification":
        model = VGG11Classifier(num_classes=NUM_CLASSES, in_channels=IN_CHANNELS,
                                dropout_p=args.dropout_p, use_bn=not args.no_bn)
    elif args.task == "localization":
        model = VGG11Localizer(in_channels=IN_CHANNELS, dropout_p=args.dropout_p,
                               image_size=args.image_size)
    else:
        model = VGG11UNet(num_classes=NUM_SEGMENTS, in_channels=IN_CHANNELS,
                          dropout_p=args.dropout_p)
 
    if args.encoder_checkpoint is not None and args.task != "classification":
        load_encoder_weights(model, args.encoder_checkpoint)
    if args.task != "classification":
        configure_freeze_strategy(model, args.freeze_strategy)
 
    return model


def build_criterion(args: argparse.Namespace) -> torch.nn.Module:
    if args.task == "classification":
        return torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    if args.task == "segmentation":
        return SegmentationLoss(num_classes=NUM_SEGMENTS)
    # localization
    if args.localization_loss == "l1":
        return torch.nn.L1Loss()
    if args.localization_loss == "mse":
        return torch.nn.MSELoss()
    if args.localization_loss == "iou":
        return CombinedLocalizationLoss(image_size=args.image_size)
    return torch.nn.SmoothL1Loss(beta=1.0)


def count_parameters(model: torch.nn.Module, trainable_only=False) -> int:
    params = (p for p in model.parameters() if p.requires_grad) if trainable_only else model.parameters()
    return sum(p.numel() for p in params)


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(dim=1)
    return torch.stack(
        [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 
        dim=1)


def box_iou(pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
    p = cxcywh_to_xyxy(pred_boxes)
    t = cxcywh_to_xyxy(target_boxes)
    tl = torch.maximum(p[:, :2], t[:, :2]) # top-left corner
    br = torch.minimum(p[:, 2:], t[:, 2:]) # bottom-right corner
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


def run_epoch(model, loader, criterion, task, device, optimizer=None, max_batches=None) -> Dict[str, float]:
    is_training = optimizer is not None
    model.train() if is_training else model.eval()
 
    total_examples = 0
    total_loss = 0.0
    cls_targets, cls_preds = [], []
    loc_iou_sum = loc_l1_sum = loc_hits50 = 0.0
    seg_correct = seg_total = seg_dice_sum = 0.0
 
    for batch_idx, batch in enumerate(loader, start=1):
        images = batch["image"].to(device=device, dtype=torch.float32)
        batch_size = images.shape[0]
        total_examples += batch_size
 
        with torch.set_grad_enabled(is_training):
            outputs = model(images)
 
            if task == "classification":
                targets = batch["class_label"].to(device=device, dtype=torch.long)
                loss = criterion(outputs, targets)
                preds = outputs.argmax(dim=1)
                cls_targets.extend(targets.detach().cpu().tolist())
                cls_preds.extend(preds.detach().cpu().tolist())
 
            elif task == "localization":
                targets = batch["bbox"].to(device=device, dtype=torch.float32)
                loss = criterion(outputs, targets)
                ious = box_iou(outputs.detach(), targets)
                loc_iou_sum += ious.sum().item()
                loc_l1_sum += torch.abs(outputs.detach() - targets).sum().item()
                loc_hits50 += (ious >= 0.5).sum().item() # number of times we get iou >= 0.5
 
            else:  # segmentation
                targets = batch["segmentation_mask"].to(device=device, dtype=torch.long)
                loss = criterion(outputs, targets)
                preds = outputs.argmax(dim=1)
                seg_correct += (preds == targets).sum().item()
                seg_total += targets.numel()
                seg_dice_sum += batch_dice_score(preds, targets, num_classes=NUM_SEGMENTS).sum().item()
 
            if is_training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
 
        total_loss += loss.item() * batch_size
        if max_batches is not None and batch_idx >= max_batches:
            break
 
    metrics = {"loss": total_loss / max(total_examples, 1)}
 
    if task == "classification":
        metrics["accuracy"] = float(np.mean(np.equal(cls_targets, cls_preds))) if cls_targets else 0.0
        metrics["macro_f1"] = float(f1_score(cls_targets, cls_preds, average="macro", zero_division=0)) if cls_targets else 0.0
 
    elif task == "localization":
        metrics["mean_iou"] = loc_iou_sum / max(total_examples, 1)
        metrics["l1_error"] = loc_l1_sum / max(total_examples * 4, 1)
        metrics["ap50_proxy"] = loc_hits50 / max(total_examples, 1)
 
    else:
        metrics["pixel_accuracy"] = seg_correct / max(seg_total, 1)
        metrics["dice"] = seg_dice_sum / max(total_examples, 1)
 
    return metrics

def maybe_init_wandb(model, args):
    if not args.use_wandb:
        return None
    if wandb is None:
        raise ImportError("wandb not installed. run without --use-wandb or pip install it")
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.group,
        name=args.run_name or f"{args.task}-{args.freeze_strategy}-ep{args.epochs}",
        config={
            "task": args.task, "epochs": args.epochs, "batch_size": args.batch_size,
            "lr": args.lr, "weight_decay": args.weight_decay, "dropout_p": args.dropout_p,
            "freeze_strategy": args.freeze_strategy, "image_size": args.image_size,
            "localization_loss": args.localization_loss,
        },
    )
    run.watch(model, log="gradients", log_freq=100)
    return run

def save_checkpoint(model: torch.nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, path)


def main() -> None:
    args = parse_args()
    args.data_dir = args.data_dir.expanduser().resolve()
    args.checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    if args.encoder_checkpoint is not None:
        args.encoder_checkpoint = args.encoder_checkpoint.expanduser().resolve()
 
    seed_everything(args.seed)
    print(f"[setup] task={args.task}  device={args.device}")
    print(f"[setup] data={args.data_dir}  checkpoints={args.checkpoint_dir}")
 
    train_loader = build_dataloader(args.data_dir, "train", args.image_size,
                                    args.batch_size, True, args.num_workers, args.device)
    val_loader = build_dataloader(args.data_dir, "val", args.image_size,
                                  args.batch_size, False, args.num_workers, args.device)
 
    model = build_model(args).to(args.device)
    criterion = build_criterion(args)
 
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("no trainable parameters after freeze strategy")
 
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
 
    print(f"[model] total={count_parameters(model):,}  trainable={count_parameters(model, trainable_only=True):,}")
 
    run = maybe_init_wandb(model, args)
 
    best_metric_name = BEST_METRIC[args.task]
    best_metric_value = float("-inf")
    checkpoint_path = args.checkpoint_dir / CHECKPOINT_NAMES[args.task]
 
    # 2.1: grab a fixed val batch once for the 2.1 activation histogram
    # fixed_images = next(iter(val_loader))["image"].to(args.device, dtype=torch.float32)
 
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = run_epoch(model, train_loader, criterion, args.task, args.device, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, args.task, args.device)
 
        # 2.1: log conv3 activation histogram every epoch (classification only)
        # if run is not None and args.task == "classification":
        #     activations = {}
        #     handle = model.encoder.bn3.register_forward_hook(
        #         lambda m, inp, out: activations.update({"conv3": out.detach().cpu()})
        #     )
        #     model.eval()
        #     with torch.no_grad():
        #         model(fixed_images)
        #     handle.remove()
        #     model.train()
        #     run.log({"epoch": epoch, "conv3_activations": wandb.Histogram(activations["conv3"].numpy().flatten())})
 
        improved = val_metrics[best_metric_name] > best_metric_value
        if improved:
            best_metric_value = val_metrics[best_metric_name]
            save_checkpoint(model, checkpoint_path)
 
        def fmt(metric):
            return ", ".join(f"{k}={v:.4f}" for k, v in sorted(metric.items()))

        print(f"[epoch {epoch:03d}/{args.epochs:03d}] train: {fmt(train_metrics)} | val: {fmt(val_metrics)} "
              f"| time={time.time()-t0:.1f}s | {'saved' if improved else 'kept'}")
        scheduler.step()
 
        if run is not None:
            run.log({
                "epoch": epoch,
                "lr": scheduler.get_last_lr()[0],
                **{f"train/{k}": v for k, v in train_metrics.items()},
                **{f"val/{k}": v for k, v in val_metrics.items()},
            })
 
    print(f"[done] best_val_{best_metric_name}={best_metric_value:.4f}")
    print(f"[done] checkpoint -> {checkpoint_path}")
 
    if run is not None:
        run.summary[f"best_val_{best_metric_name}"] = best_metric_value
        run.summary["checkpoint_path"] = str(checkpoint_path)
        run.finish()

if __name__ == "__main__":
    main()
