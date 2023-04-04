import torch
import torch.nn.functional as F


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = torch.Tensor(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def ssim_l1_loss(
    img1, img2, window_size=11, alpha=0.84, beta=0.12, gamma=0.04, L1_weight=1.0
):
    if not img1.size() == img2.size():
        raise ValueError("Input images must have the same dimensions.")

    window = create_window(window_size, img1.size()[1])
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    mu1 = F.conv2d(
        img1, window, stride=1, padding=window_size // 2, groups=img1.size()[1]
    )
    mu2 = F.conv2d(
        img2, window, stride=1, padding=window_size // 2, groups=img1.size()[1]
    )

    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(
            img1 * img1,
            window,
            stride=1,
            padding=window_size // 2,
            groups=img1.size()[1],
        )
        - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(
            img2 * img2,
            window,
            stride=1,
            padding=window_size // 2,
            groups=img1.size()[1],
        )
        - mu2_sq
    )
    sigma12 = (
        F.conv2d(
            img1 * img2,
            window,
            stride=1,
            padding=window_size // 2,
            groups=img1.size()[1],
        )
        - mu1_mu2
    )

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    ssim = ssim_map.mean()

    L1_loss = F.l1_loss(img1, img2)

    loss = alpha * (1 - ssim) + beta * L1_loss + gamma * (1 - ssim_map)

    return L1_weight * loss
