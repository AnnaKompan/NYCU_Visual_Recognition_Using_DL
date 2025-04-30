import os

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import tifffile as tiff


class CustomDataset(Dataset):
    def __init__(self, image_paths, mask_paths):
        self.image_paths = image_paths
        self.mask_paths = mask_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = np.array(img)
        H, W = img.shape[:2]

        masks, labels, boxes = [], [], []

        for mask_path in self.mask_paths[idx]:
            mask_arr = tiff.imread(mask_path)
            cls = int(
                os.path.basename(mask_path)
                .replace("class", "")
                .replace(".tif", "")
            )
            for inst_id in np.unique(mask_arr):
                if inst_id == 0:
                    continue
                binary = (mask_arr == inst_id)
                if not binary.any():
                    continue

                masks.append(torch.as_tensor(binary, dtype=torch.uint8))
                labels.append(cls)
                ys, xs = np.where(binary)
                xmin, xmax = xs.min(), xs.max()
                ymin, ymax = ys.min(), ys.max()
                boxes.append([xmin, ymin, xmax, ymax])

        if len(masks) == 0:
            masks_tensor = torch.zeros((0, H, W), dtype=torch.uint8)
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
        else:
            masks_tensor = torch.stack(masks)
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)

        target = {
            'boxes': boxes_tensor,
            'labels': labels_tensor,
            'masks': masks_tensor,
            'image_id': torch.tensor(
                [idx],
                dtype=torch.int64
            )
        }

        img_tensor = torch.as_tensor(
            img,
            dtype=torch.float32
        ).permute(2, 0, 1) / 255.0

        return img_tensor, target
