import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image



def load_image_raw(path, device):
    """
    Returns:
      img_cpu : PIL.Image
      x       : torch.Tensor (1, 3, H, W), values in [0,1]
    """
    img_cpu = Image.open(path).convert("RGB")
    x = transforms.ToTensor()(img_cpu).unsqueeze(0).to(device)
    return img_cpu, x


def draw_mask(mask):
    if torch.is_tensor(mask):
        mask = mask.cpu().squeeze().numpy()
    plt.imshow(mask, cmap="gray")
    plt.axis("off")
    plt.show()

def draw_masked_image(img, mask):
    if not isinstance(img, np.ndarray):
        img = np.array(img)

    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0

    if torch.is_tensor(mask):
        mask = mask.cpu().permute(1, 2, 0).numpy()  # (H, W, 1)

    masked = img * mask

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    # left: mask
    axes[0].imshow(mask.squeeze(), cmap="gray")
    axes[0].set_title("Mask")
    axes[0].axis("off")

    # right: masked image
    axes[1].imshow(masked)
    axes[1].set_title("Masked image")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


