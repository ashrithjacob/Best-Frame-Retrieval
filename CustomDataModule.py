import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from processing import imshow, imexpl, grey_to_rgb

class CustomDataModule(pl.LightningDataModule):
    def __init__(self, data_dir="./video_frame", batch_size=32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def prepare_data(self):
        # download or prepare data if needed
        # leave it blank if data is already prepared
        pass

    def transform(self, stage=None):
        # define transforms
        transform = transforms.Compose(
        [   transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.Resize((96, 192), antialias=True),
            transforms.Lambda(grey_to_rgb),
        ]
    )
        # define dataset
        self.dataset = ImageFolder(self.data_dir, transform=transform)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size)

if __name__ == '__main__':
    dm = CustomDataModule(data_dir='./videos/video_data/', batch_size=2)
    dm.transform()
    print(dm.dataset)
    print(len(dm.dataset))
    print(dm.dataset[0][0].shape)
    print(dm.dataset[0][1])
    train_dataloader = dm.train_dataloader()
    #train_dataloader = DataLoader(dm.dataset, batch_size=2, shuffle=True)
    dataiter = iter(train_dataloader)
    print(len(dataiter))
    images,i = next(dataiter)
    imshow(torchvision.utils.make_grid(images))
    imexpl(images)
    print(i)