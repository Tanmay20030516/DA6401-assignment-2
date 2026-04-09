"""Dataset for Oxford-IIIT Pet."""

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

    Directory layout expected:
        root_dir/
            images/*.jpg
            annotations/
                list.txt          (trainval metadata with class ids)
                trainval.txt      (trainval image names)
                test.txt          (test image names, no bboxes)
                trimaps/*.png
                xmls/*.xml

    Splits:
        train / val  — derived from trainval.txt (90% / 10% stratified by class)
        test         — from test.txt (no bbox annotations; those fields = not_found)
    """

    # trimap pixel values -> class indices (background=0, body=1, boundary=2)
    trimap_to_class = {1: 1, 2: 0, 3: 2}
    val_size = 0.1
    seed = 42

    def __init__(self, root_dir: str, split: str = "train", image_size: int = 224):
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size

        manifest_path = os.path.join(root_dir, "manifest.csv")
        if os.path.exists(manifest_path):
            full_df = pd.read_csv(manifest_path)
        else:
            full_df = self._build_manifest(root_dir)
            full_df.to_csv(manifest_path, index=False)
            print(f"manifest saved to {manifest_path}")

        self.df = full_df[full_df["split"] == split].reset_index(drop=True)
        print(f"split='{split}' — {len(self.df)} samples")

        self.transform = self._build_transforms(split, image_size)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns a dict with:
            image : float32 [3, H, W], pixels in [0, 1]
            class_label : int64 scalar, 0-indexed breed id
            bbox : float32 [4], pixel-space (cx, cy, w, h); fallback = full image
            segmentation_mask : int64 [H, W], values in {0=bg, 1=body, 2=boundary}
        """
        row = self.df.iloc[idx]

        image_np = np.array(
            Image.open(row["image_path"]).convert("RGB"), dtype=np.uint8
        )
        mask_np = self._read_trimap(row["mask_path"]).astype(np.uint8)

        # bbox stored as normalised [xmin, ymin, xmax, ymax] in manifest
        xmin = float(row["xmin"]) if str(row["xmin"]) != "not_found" else 0.0
        ymin = float(row["ymin"]) if str(row["ymin"]) != "not_found" else 0.0
        xmax = float(row["xmax"]) if str(row["xmax"]) != "not_found" else 1.0
        ymax = float(row["ymax"]) if str(row["ymax"]) != "not_found" else 1.0

        result = self.transform(
            image=image_np,
            mask=mask_np,
            bboxes=[(xmin, ymin, xmax, ymax)],
            bbox_labels=[0],
        )

        image_out = result["image"]
        mask_out = result["mask"]

        if result["bboxes"]:
            xmin, ymin, xmax, ymax = result["bboxes"][0]
        else:
            xmin, ymin, xmax, ymax = 0.0, 0.0, 1.0, 1.0

        # normalised -> pixel-space centre-format
        cx = (xmin + xmax) / 2.0 * self.image_size
        cy = (ymin + ymax) / 2.0 * self.image_size
        w = (xmax - xmin) * self.image_size
        h = (ymax - ymin) * self.image_size
        bbox_out = np.array([cx, cy, w, h], dtype=np.float32)

        return {
            "image": torch.from_numpy(image_out).permute(2, 0, 1),
            "class_label": torch.tensor(int(row["class_id"]), dtype=torch.long),
            "bbox": torch.from_numpy(bbox_out),
            "segmentation_mask": torch.from_numpy(mask_out.astype(np.int64)),
        }

    @classmethod
    def _build_manifest(cls, root_dir: str) -> pd.DataFrame:
        """
        Build the manifest CSV once from trainval.txt and test.txt.

        Reads class ids from list.txt (which covers all images).
        Bboxes for test images are marked not_found because test.txt images
        have no XML annotations.
        """
        annot_dir = os.path.join(root_dir, "annotations")
        image_dir = os.path.join(root_dir, "images")
        trimap_dir = os.path.join(annot_dir, "trimaps")
        xml_dir = os.path.join(annot_dir, "xmls")

        # build a name -> class_id lookup from list.txt
        class_id_for = {}
        with open(os.path.join(annot_dir, "list.txt")) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                class_id_for[parts[0]] = int(parts[1]) - 1  # 0-indexed

        trainval_rows = cls._parse_split_file(
            os.path.join(annot_dir, "trainval.txt"),
            class_id_for,
            image_dir,
            trimap_dir,
            xml_dir,
            has_bbox=True,
        )
        test_rows = cls._parse_split_file(
            os.path.join(annot_dir, "test.txt"),
            class_id_for,
            image_dir,
            trimap_dir,
            xml_dir,
            has_bbox=False,
        )

        trainval_df = pd.DataFrame(trainval_rows)
        test_df = pd.DataFrame(test_rows)
        test_df["split"] = "test"

        # stratified 90/10 split on trainval
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=cls.val_size, random_state=cls.seed
        )
        train_idx, val_idx = next(sss.split(trainval_df, trainval_df["class_id"]))

        train_df = trainval_df.iloc[train_idx].copy()
        val_df = trainval_df.iloc[val_idx].copy()
        train_df["split"] = "train"
        val_df["split"] = "val"

        df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        return df

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
        """Read image names from a split file and build row dicts."""
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
                    xmin, ymin, xmax, ymax = cls._parse_bbox(xml_path)
                    xml_col = xml_path if os.path.exists(xml_path) else "not_found"
                else:
                    xmin = ymin = xmax = ymax = "not_found"
                    xml_col = "not_found"

                rows.append(
                    {
                        "image_path": image_path,
                        "mask_path": mask_path,
                        "xml_path": xml_col,
                        "class_id": class_id,
                        "xmin": xmin,
                        "ymin": ymin,
                        "xmax": xmax,
                        "ymax": ymax,
                        "split": "temp",
                    }
                )
        return rows

    @staticmethod
    def _build_transforms(split: str, image_size: int) -> A.Compose:
        """
        Training: spatial + colour augmentations, then scale to [0, 1].
        Val / test: resize + scale only.
        """
        bbox_params = A.BboxParams(
            format="albumentations",  # normalised [x0,y0,x1,y1]
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
        else:
            return A.Compose(
                [
                    A.Resize(image_size, image_size),
                    A.ToFloat(max_value=255.0),
                ],
                bbox_params=bbox_params,
            )

    @staticmethod
    def _parse_bbox(xml_path: str) -> tuple:
        """
        Read bounding box from a PASCAL-VOC XML and normalise by image dims.
        Falls back to (0, 0, 1, 1) when the file is missing.
        """
        if not os.path.exists(xml_path):
            return (0.0, 0.0, 1.0, 1.0)

        root = ET.parse(xml_path).getroot()
        size = root.find("size")
        orig_w = float(size.find("width").text)  # type: ignore
        orig_h = float(size.find("height").text)  # type: ignore

        bndbox = root.find(".//bndbox")
        xmin = float(bndbox.find("xmin").text) / orig_w  # type: ignore
        ymin = float(bndbox.find("ymin").text) / orig_h  # type: ignore
        xmax = float(bndbox.find("xmax").text) / orig_w  # type: ignore
        ymax = float(bndbox.find("ymax").text) / orig_h  # type: ignore

        xmin, xmax = sorted([np.clip(xmin, 0.0, 1.0), np.clip(xmax, 0.0, 1.0)])
        ymin, ymax = sorted([np.clip(ymin, 0.0, 1.0), np.clip(ymax, 0.0, 1.0)])
        return (xmin, ymin, xmax, ymax)

    @classmethod
    def _read_trimap(cls, mask_path: str) -> np.ndarray:
        """
        Load trimap PNG and remap values: 1->1 (body), 2->0 (bg), 3->2 (boundary).
        Returns HW int32 array. Falls back to a 1x1 background array if missing.
        """
        if not os.path.exists(mask_path):
            return np.zeros((1, 1), dtype=np.int32)

        # raw = np.array(Image.open(mask_path), dtype=np.int32)
        raw = np.array(
            Image.open(mask_path).convert("L"), dtype=np.int32
        )  # read as grayscale (all 3 channels have same values only in the trimap)
        mask = np.zeros_like(raw)
        for src, dst in cls.trimap_to_class.items():
            mask[raw == src] = dst
        return mask
