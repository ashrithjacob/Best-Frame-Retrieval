import torch
from torch import nn


# Define the autoencoder architecture
class Autoencoder(nn.Module):
    def __init__(self, get_latent=False):
        super(Autoencoder, self).__init__()
        self.get_latent = get_latent
        # Activation functions
        self.relu = nn.LeakyReLU(0.01)
        self.tanh = nn.Tanh()
        # Encoder
        self.bn0 = nn.BatchNorm2d(3, momentum=0.5)
        self.conv1 = nn.Conv2d(
            3, 32, kernel_size=3, stride=1, padding=1
        )  # 96*192 -> 96*192
        self.bn1 = nn.BatchNorm2d(32, momentum=0.5)
        self.conv2 = nn.Conv2d(
            32, 64, kernel_size=3, stride=1, padding=1
        )  # 96*192 -> 96*192
        self.bn2 = nn.BatchNorm2d(64, momentum=0.5)
        self.conv3 = nn.Conv2d(
            64, 128, kernel_size=4, stride=2, padding=1
        )  # 96*192 -> 48*96
        self.bn3 = nn.BatchNorm2d(128, momentum=0.5)
        self.conv4 = nn.Conv2d(
            128, 64, kernel_size=4, stride=2, padding=1
        )  # 48*96 -> 24*48
        self.bn4 = nn.BatchNorm2d(64, momentum=0.5)
        self.conv5 = nn.Conv2d(
            64, 8, kernel_size=4, stride=2, padding=1
        )  # 24*48 -> 12*24
        self.bn5 = nn.BatchNorm2d(8, momentum=0.5)
        self.encoder = nn.Sequential(
            self.bn0,
            self.conv1,
            self.bn1,
            self.relu,
            self.conv2,
            self.bn2,
            self.relu,
            self.conv3,
            self.bn3,
            self.relu,
            self.conv4,
            self.bn4,
            self.relu,
            self.conv5,
            self.bn5,
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
            self.bn4,
            self.relu,
            self.t_conv2,
            self.bn3,
            self.relu,
            self.t_conv3,
            self.bn2,
            self.relu,
            self.t_conv4,
            self.bn1,
            self.relu,
            self.t_conv5,
            self.tanh,
        )

    def forward(self, x):
        # Encoder
        x = self.encoder(x)
        latent = x
        x = self.decoder(x)
        if self.get_latent:
            return latent
        else:
            return x
