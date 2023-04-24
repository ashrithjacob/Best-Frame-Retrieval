import torch
import time
import os
import matplotlib.pyplot as plt
import torchvision
from torch import nn
from processing import checkpoint, imshow, resume, get_epoch


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


# Define the training function
def train(model, device, train_loader, optimizer, criterion, start_epoch, end_epoch, checkpoint_dir):
    model.train()
    start_time = time.time()
    for epoch in range(start_epoch, end_epoch + 1):
        epoch_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            output = model(data)
            loss = criterion(data, output)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if batch_idx % 100 == 0:
                print(
                    f"Epoch [{epoch}/{end_epoch}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}"
                )
        print(f'average loss for epoch {epoch} is {epoch_loss/len(train_loader):.4f}')
        if epoch % 10 == 0:
            checkpoint(model, f"{checkpoint_dir+'/'}MS_SSIM_L1-epoch-{e}.pth")
    print("Finished Training in time", (time.time() - start_time) / 60, "mins")


def test(model, device, test_loader, criterion, display_img=False):
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(test_loader):
            data = data.to(device)
            output = model(data)
            loss = criterion(data, output)
            if (batch_idx+1) % 5 == 0 and display_img:
                print (f"Input image(ABOVE) vs Reconstructed image(BELOW), for batch:[{batch_idx+1}/{len(test_loader)}]")
                imshow(torchvision.utils.make_grid(data.to("cpu")))
                imshow(torchvision.utils.make_grid(output.to("cpu")))
                print(f"Loss: {loss.item():.4f}")
                print("\n")


def disp_test_loss(model, device, test_loader, criterion, checkpoint_dir):
    checkpoint_loss =[]
    epoch_number = []
    checkpoints = [f for f in os.listdir(checkpoint_dir) if not f.startswith('.')]
    print(f' checkpoints found: {checkpoints}')
    # running the model on evaluation mode
    model.eval()
    for c in checkpoints:
        resume(model, f"{str(checkpoint_dir)}/{c}")
        c_loss = 0
        for (data, _ )in test_loader:
            with torch.no_grad():
                data = data.to(device)
                output = model(data)
                loss = criterion(data, output)
                c_loss += loss.item()
                checkpoint_loss.append(c_loss/len(test_loader))
                epoch_number.append(get_epoch(c))
    plt.scatter(epoch_number, checkpoint_loss)
    plt.xlabel("epoch number")
    plt.ylabel("Average test loss")
    plt.show()
