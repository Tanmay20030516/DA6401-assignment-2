import os
import xml.etree.ElementTree as ET

import albumentations as A
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
import torch
from torch.utils.data import Dataset


class OxfordIIITPetDataset(Dataset):
    """
    Oxford-IIIT Pet dataset for classification, localization, and segmentation.
    """

    # trimap pixel values -> class indices (background=0, body=1, boundary=2)
    trimap_to_class = {1: 1, 2: 0, 3: 2}
    val_size = 0.1
    seed = 42

    def __init__(self, root_dir: str, split: str = "train", image_size: int = 224):
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size

        self.df = self._load_split(root_dir, split)
        print(f"split='{split}' — {len(self.df)} samples")

        self.transform = self._build_transforms(split, image_size)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns:
            image              : float32 tensor [3, H, W], pixels in [0, 1]
            class_label        : int64 scalar, 0-indexed breed id
            bbox               : float32 [4], pixel-space (cx, cy, w, h)
            segmentation_mask  : int64 [H, W], values in {0=bg, 1=body, 2=boundary}
        """
        row = self.df.iloc[idx]

        image_np = np.array(
            Image.open(row["image_path"]).convert("RGB"), dtype=np.uint8
        )
        mask_np = self._read_trimap(row["mask_path"]).astype(np.uint8)

        # bbox is stored normalised [xmin, ymin, xmax, ymax] (albumentations format)
        xmin = float(row["xmin"])
        ymin = float(row["ymin"])
        xmax = float(row["xmax"])
        ymax = float(row["ymax"])

        result = self.transform(
            image=image_np,
            mask=mask_np,
            bboxes=[(xmin, ymin, xmax, ymax)],
            bbox_labels=[0],
        )

        image_out = result["image"]  # HWC float32 in [0,1] from A.ToFloat
        mask_out = result["mask"]  # HW uint8

        if result["bboxes"]:
            xmin, ymin, xmax, ymax = result["bboxes"][0]
        else:
            xmin, ymin, xmax, ymax = 0.0, 0.0, 1.0, 1.0

        # normalised -> pixel-space centre format
        cx = ((xmin + xmax) / 2.0) * self.image_size
        cy = ((ymin + ymax) / 2.0) * self.image_size
        w = (xmax - xmin) * self.image_size
        h = (ymax - ymin) * self.image_size

        return {
            # permute HWC -> CHW as expected by PyTorch
            "image": torch.from_numpy(image_out).permute(2, 0, 1).float(),
            "class_label": torch.tensor(int(row["class_id"]), dtype=torch.long),
            "bbox": torch.tensor([cx, cy, w, h], dtype=torch.float32),
            "segmentation_mask": torch.from_numpy(mask_out.astype(np.int64)),
        }

    @classmethod
    def _load_split(cls, root_dir: str, split: str) -> pd.DataFrame:
        annot_dir = os.path.join(root_dir, "annotations")
        image_dir = os.path.join(root_dir, "images")
        trimap_dir = os.path.join(annot_dir, "trimaps")
        xml_dir = os.path.join(annot_dir, "xmls")

        class_id_for = cls._read_class_ids(annot_dir)

        if split in ("train", "val"):
            rows = cls._parse_split_file(
                os.path.join(annot_dir, "trainval.txt"),
                class_id_for,
                image_dir,
                trimap_dir,
                xml_dir,
                has_bbox=True,
            )
            df = pd.DataFrame(rows)

            sss = StratifiedShuffleSplit(
                n_splits=1, test_size=cls.val_size, random_state=cls.seed
            )
            train_idx, val_idx = next(sss.split(df, df["class_id"]))
            idx = train_idx if split == "train" else val_idx
            return df.iloc[idx].reset_index(drop=True)

        # test split
        rows = cls._parse_split_file(
            os.path.join(annot_dir, "test.txt"),
            class_id_for,
            image_dir,
            trimap_dir,
            xml_dir,
            has_bbox=False,
        )
        return pd.DataFrame(rows).reset_index(drop=True)

    @staticmethod
    def _read_class_ids(annot_dir: str) -> dict:
        class_id_for = {}
        with open(os.path.join(annot_dir, "list.txt")) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                class_id_for[parts[0]] = int(parts[1]) - 1  # 0-indexed
        return class_id_for

    @classmethod
    def _parse_split_file(
        cls,
        split_file: str,
        class_id_for: dict,
        image_dir: str,
        trimap_dir: str,
        xml_dir: str,
        has_bbox: bool,
    ) -> list:
        rows = []
        with open(split_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                name = line.split()[0]

                image_path = os.path.join(image_dir, name + ".jpg")
                mask_path = os.path.join(trimap_dir, name + ".png")
                xml_path = os.path.join(xml_dir, name + ".xml")
                class_id = class_id_for.get(name, -1)

                if has_bbox:
                    # skip images without a parseable bounding box XML
                    if not os.path.exists(xml_path):
                        continue
                    bbox = cls._parse_bbox(xml_path)
                    if bbox is None:
                        continue
                    xmin, ymin, xmax, ymax = bbox
                else:
                    # test images: full-image fallback in normalised coords
                    xmin, ymin, xmax, ymax = 0.0, 0.0, 1.0, 1.0

                rows.append(
                    {
                        "image_path": image_path,
                        "mask_path": mask_path,
                        "xml_path": xml_path if has_bbox else "not_found",
                        "class_id": class_id,
                        "xmin": xmin,
                        "ymin": ymin,
                        "xmax": xmax,
                        "ymax": ymax,
                    }
                )
        return rows

    @staticmethod
    def _build_transforms(split: str, image_size: int) -> A.Compose:
        bbox_params = A.BboxParams(
            format="albumentations",
            label_fields=["bbox_labels"],
            min_visibility=0.2,
            clip=True,
        )

        if split == "train":
            return A.Compose(
                [
                    A.Resize(image_size, image_size),
                    A.HorizontalFlip(p=0.5),
                    A.Rotate(limit=10, border_mode=0, p=0.4),
                    A.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05, p=0.5
                    ),
                    A.GaussianBlur(blur_limit=2, p=0.2),
                    A.ToFloat(max_value=255.0),
                ],
                bbox_params=bbox_params,
            )

        return A.Compose(
            [
                A.Resize(image_size, image_size),
                A.ToFloat(max_value=255.0),
            ],
            bbox_params=bbox_params,
        )

    @staticmethod
    def _parse_bbox(xml_path: str):
        try:
            root = ET.parse(xml_path).getroot()
            size = root.find("size")
            orig_w = float(size.find("width").text)  # type: ignore
            orig_h = float(size.find("height").text)  # type: ignore
            bndbox = root.find(".//bndbox")
            xmin = float(bndbox.find("xmin").text) / orig_w  # type: ignore
            ymin = float(bndbox.find("ymin").text) / orig_h  # type: ignore
            xmax = float(bndbox.find("xmax").text) / orig_w  # type: ignore
            ymax = float(bndbox.find("ymax").text) / orig_h  # type: ignore
        except Exception:
            return None

        xmin, xmax = sorted([np.clip(xmin, 0.0, 1.0), np.clip(xmax, 0.0, 1.0)])
        ymin, ymax = sorted([np.clip(ymin, 0.0, 1.0), np.clip(ymax, 0.0, 1.0)])

        # degenerate box — treat as missing
        if xmax <= xmin or ymax <= ymin:
            return None

        return xmin, ymin, xmax, ymax

    @classmethod
    def _read_trimap(cls, mask_path: str) -> np.ndarray:
        if not os.path.exists(mask_path):
            return np.zeros((1, 1), dtype=np.int32)

        raw = np.array(Image.open(mask_path).convert("L"), dtype=np.int32)
        mask = np.zeros_like(raw)
        for src, dst in cls.trimap_to_class.items():
            mask[raw == src] = dst
        return mask
