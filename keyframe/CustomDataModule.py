import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from processing import imshow, imexpl, grey_to_rgb


class CustomDataModule(pl.LightningDataModule):
    def __init__(self, data_dir="./VIDEOS/video_data/", batch_size=32, train_size=0.8):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_size = train_size

    def prepare_data(self):
        # download or prepare data if needed
        # Add Extract frames call here
        pass

    def transform(self, stage=None):
        # define transforms
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
                transforms.Resize((96, 192), antialias=True),
                transforms.Lambda(grey_to_rgb),
            ]
        )
        # define dataset
        self.dataset = ImageFolder(self.data_dir, transform=transform)

    def split(self):
        # split train and test dataset
        random_seed = torch.Generator().manual_seed(42)
        self.train, self.test = torch.utils.data.random_split(
            self.dataset,
            [self.train_size, 1.0 - self.train_size],
            generator=random_seed,
        )

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True)

    def ordered_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=True)


if __name__ == "__main__":
    dm = CustomDataModule(data_dir="./VIDEOS/sim_pairs/", batch_size=2, train_size=0.8)
    dm.transform()
    # dm.dataset is a 2d tuple containing the whole dataset:
    #   - The first dimension is the index of the image(ordered by the folder name)
    #   - The second dimension is the image tensor(at 0th index) and its label(at 1st index)
    #   - The image itself is a 3d tensor of size (3, 96, 192)
    print(dm.dataset)
    I = 10
    print(f"Image tensor shape of {I}th image: {dm.dataset[I][0].shape}")
    print(f"Image class label of {I}th index: {dm.dataset[I][1]}")
    print(f"number of clases: {len(dm.dataset.classes)}")
    #  train dataloader (shuffled):
    """
    train_dataloader = dm.train_dataloader()
    dataiter = iter(train_dataloader)
    print(len(dataiter))
    images,i = next(dataiter)
    imshow(torchvision.utils.make_grid(images))
    imexpl(images)
    print(i)
    """
    # Ordered dataloader:
    ordered_dataloader = dm.ordered_dataloader()
    dataiter = iter(ordered_dataloader)
    for images, i in dataiter:
        imshow(torchvision.utils.make_grid(images))
        imexpl(images)
        print("class", i)
