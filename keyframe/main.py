import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import time
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from processing import grey_to_rgb
from MS_SSIM_L1_loss import MS_SSIM_L1_LOSS
from model import Autoencoder

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
    criterion = MS_SSIM_L1_LOSS()
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
            if (i + 1) % 100 == 0:
                print(
                    f"Epoch [{e+1}/{epoch}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}"
                )
    print("Finished Training in time", (time.time() - start_time) / 60, "mins")
    torch.save(model.state_dict(), "autoencoder_MS_SSIM_L1.pth") 
