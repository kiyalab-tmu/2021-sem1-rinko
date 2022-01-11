import torch
import torchvision.transforms as transforms
from torchvision.datasets import  mnist,cifar
import torch_fidelity
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class ImageFolder(Dataset):
    IMG_EXTENSIONS = [".png"]

    def __init__(self, img_dir, transform=None):
        # 画像ファイルのパス一覧を取得する。
        self.img_paths = self._get_img_paths(img_dir)
        self.transform = transform

    def __getitem__(self, index):
        path = self.img_paths[index]

        # 画像を読み込む。
        img = Image.open(path)
        img = np.uint8(img)
        if self.transform is not None:
            img = self.transform(img)
        img = torch.from_numpy(img.astype(np.uint8))
        img = img.permute(2,0,1)
        return img

    def _get_img_paths(self, img_dir):
        img_dir = Path(img_dir)
        img_paths = [
            p for p in img_dir.iterdir() if p.suffix in ImageFolder.IMG_EXTENSIONS
        ]

        return img_paths

    def __len__(self):
        return len(self.img_paths)

# Dataset を作成する。
dataset_fake = ImageFolder("./default-cifar10/genereated_images/499")

# metric計算
metrics = torch_fidelity.calculate_metrics(
            input1=dataset_fake,
            input2='cifar10-train',
            isc=True,
            fid=True,
            kid=True,
            kid_subset_size = 50,
            batch_size = 50,
        )

print(metrics)