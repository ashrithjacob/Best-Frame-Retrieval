import os
import pandas as pd
import torch
import torchvision
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision import transforms
from processing import grey_to_rgb, imshow, imexpl

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = cv2.imread(img_path, -1)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


if __name__ == "__main__":
    transform = transforms.Compose(
        [   transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.Resize((96, 192), antialias=True),
            transforms.Lambda(grey_to_rgb),
        ]
    )

    dataset = CustomImageDataset(
        annotations_file="./data/labels.csv",
        img_dir="./data/data2/",
        transform=transform,
    )

    train_dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    dataiter = iter(train_dataloader)
    images,_ = next(dataiter)
    imshow(torchvision.utils.make_grid(images))
    imexpl(images)