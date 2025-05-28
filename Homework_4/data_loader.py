import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class CustomDataset(Dataset):
    def __init__(self, degraded_dir, clean_dir,  file_list=None, transform=None):
        self.degraded_dir = degraded_dir
        self.clean_dir = clean_dir
        if file_list is not None:
          self.filenames = sorted(file_list)
        else:
          self.filenames = sorted([
            f for f in os.listdir(degraded_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        self.transform = transform if transform else T.ToTensor()

    def __getitem__(self, idx):
        degraded_path = os.path.join(
            self.degraded_dir,
            self.filenames[idx]
        )
        clean_name = self.filenames[idx]
        if 'rain' in clean_name:
            clean_name = clean_name.replace(
                '.png', ''
                ).replace('rain-', 'rain_clean-') + '.png'
        elif 'snow' in clean_name:
            clean_name = clean_name.replace(
                '.png', ''
                ).replace('snow-', 'snow_clean-') + '.png'
        else:
            raise ValueError("Unknown filename format: " + clean_name)

        clean_path = os.path.join(self.clean_dir, clean_name)

        degraded_img = Image.open(degraded_path).convert("RGB")
        clean_img = Image.open(clean_path).convert("RGB")

        degraded_img = self.transform(degraded_img)
        clean_img = self.transform(clean_img)

        return degraded_img, clean_img

    def __len__(self):
        return len(self.filenames)
