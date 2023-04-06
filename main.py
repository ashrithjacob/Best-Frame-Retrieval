import torch
import torchvision
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from processing import grey_to_rgb, ms_ssim_loss
import matplotlib.pyplot as plt
import numpy as np
import time

# Define the autoencoder architecture
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # Activation functions
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        # Encoder
        self.conv1 = nn.Conv2d(
            3, 32, kernel_size=3, stride=1, padding=1
        )  # 96*192 -> 96*192
        self.conv2 = nn.Conv2d(
            32, 64, kernel_size=3, stride=1, padding=1
        )  # 96*192 -> 96*192
        self.conv3 = nn.Conv2d(
            64, 128, kernel_size=4, stride=2, padding=1
        )  # 96*192 -> 48*96
        self.conv4 = nn.Conv2d(
            128, 64, kernel_size=4, stride=2, padding=1
        )  # 48*96 -> 24*48
        self.conv5 = nn.Conv2d(
            64, 8, kernel_size=4, stride=2, padding=1
        )  # 24*48 -> 12*24
        self.encoder = nn.Sequential(
            self.conv1,
            self.relu,
            self.conv2,
            self.relu,
            self.conv3,
            self.relu,
            self.conv4,
            self.relu,
            self.conv5,
            self.relu,
        )

        # Decoder
        self.t_conv1 = nn.ConvTranspose2d(
            8, 64, kernel_size=4, stride=2, padding=1
        )  # 12*24 -> 24*48
        self.t_conv2 = nn.ConvTranspose2d(
            64, 128, kernel_size=4, stride=2, padding=1
        )  # 24*48 -> 48*96
        self.t_conv3 = nn.ConvTranspose2d(
            128, 64, kernel_size=4, stride=2, padding=1
        )  # 48*96 -> 96*192
        self.t_conv4 = nn.ConvTranspose2d(
            64, 32, kernel_size=3, stride=1, padding=1
        )  # 96*192 -> 96*192
        self.t_conv5 = nn.ConvTranspose2d(
            32, 3, kernel_size=3, stride=1, padding=1
        )  # 96*192 -> 96*192
        self.decoder = nn.Sequential(
            self.t_conv1,
            self.relu,
            self.t_conv2,
            self.relu,
            self.t_conv3,
            self.relu,
            self.t_conv4,
            self.relu,
            self.t_conv5,
            self.tanh,
        )

    def forward(self, x):
        # Encoder
        x = self.encoder(x)
        x = self.decoder(x)
        return x


if __name__ == "__main__":
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load the dataset
    transform = transforms.Compose(
        [   transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.Resize((96, 192), antialias=True),
            transforms.Lambda(grey_to_rgb),
        ]
    )
    trainset = datasets.Caltech256(
        "./DATA",
        download=True,
        transform=transform,
    )
    # random seed and batch size
    random_seed = torch.Generator().manual_seed(42)
    batch_size = 4
    # split train and test
    train, test = torch.utils.data.random_split(trainset, [0.8, 0.2], generator=random_seed)
    # dataloader
    dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    trainloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True)
    # check the shape of the data
    trainiter = iter(trainloader)
    img, label = next(trainiter)
    print(img.shape, label.shape) # torch.Size([64, 3, 96, 192]) torch.Size([64])
    # Setting parameters
    model = Autoencoder()
    model = model.to(device)
    criterion = nn.MSELoss()
    learning_rate = 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    epoch = 10
    n_total_steps = len(trainloader)
    # Training
    start_time = time.time()
    for e in range(epoch):
        for i, (img_in, _) in enumerate(trainloader):
            img_in = img_in.to(device)
            optimizer.zero_grad()
            img_out = model(img_in)
            loss = criterion(img_in, img_out)
            loss.backward()
            optimizer.step()
            if (i + 1) % 200 == 0:
                print(
                    f"Epoch [{e+1}/{epoch}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}"
                )
    print("Finished Training in time", (time.time() - start_time) / 60, "mins")
    torch.save(model.state_dict(), "autoencoder.pth") 
