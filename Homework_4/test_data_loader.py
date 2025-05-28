import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class TestDataset(Dataset):
    def __init__(self, degraded_img, transform=None):
        self.degraded_img = degraded_img
        self.filenames = sorted([
            f for f in os.listdir(degraded_img)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        self.transform = transform if transform else T.ToTensor()

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        img_path = os.path.join(self.degraded_img, filename)
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, filename

    def __len__(self):
        return len(self.filenames)
