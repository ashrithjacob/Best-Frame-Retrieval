import torch
import torchvision
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from processing import grey_to_rgb, imshow, imexpl

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
    """
    valset = datasets.Caltech256(
        "./DATA",
        download=True,
        train=False,
        transform=transform,
    )
    """
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    #valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

    # get some random training images
    dataiter = iter(trainloader)
    images, _ = next(dataiter)

    # show images
    imshow(torchvision.utils.make_grid(images))

    # explore the dataset
    imexpl(images)
    """
    # Create an instance of the autoencoder model
    model = Autoencoder()

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Load the dataset
    dataset = DataLoader(...)

    # Train the autoencoder
    num_epochs = 10
    for epoch in range(num_epochs):
        running_loss = 0.0
        for data in dataset:
            inputs, _ = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print("Epoch [%d/%d], Loss: %.4f" % (epoch + 1, num_epochs, running_loss))

    # Save the trained model
    torch.save(model.state_dict(), "autoencoder.pth")
"""
