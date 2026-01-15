from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm


ROOT = Path(__file__).parent.parent.parent
IMAGES = ROOT / "images"
ARTIFACTS = ROOT / "artifacts"

def resize_mask(mask, up_size):
    mask_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)  # (1,1,s,s)
    mask_up = F.interpolate(mask_t, size=up_size.tolist(), mode='bilinear', align_corners=False)
    return mask_up.squeeze()

class RISE(nn.Module):
    def __init__(self, model, input_size, gpu_batch=100, device=None):
        super(RISE, self).__init__()
        self.model = model
        self.input_size = input_size
        self.gpu_batch = gpu_batch
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def generate_masks(self, N, s, p, savepath = ARTIFACTS / 'masks.npy'):
        cell_size = np.ceil(np.array(self.input_size) / s).astype(int)
        up_size = (s + 1) * cell_size   # +1 so we can shift grids

        grids = np.random.rand(N, s, s) < p
        grids = grids.astype('float32')

        self.masks = np.empty((N, *self.input_size), dtype=np.float32)

        for i in tqdm(range(N), desc='Generating filters'):
            # Random shifts
            x = np.random.randint(0, cell_size[0])
            y = np.random.randint(0, cell_size[1])

            up_mask = resize_mask(grids[i], up_size)
            self.masks[i, :, :] = up_mask[x:x + self.input_size[0],
                                        y:y + self.input_size[1]]

        self.masks = self.masks.reshape(N, 1, *self.input_size)    # (N, 1, H, W) add dim for channel
        np.save(savepath, self.masks)
        self.masks = torch.from_numpy(self.masks)
        self.masks = self.masks.to(self.device)
        self.N = N
        self.p = p

    def load_masks(self, filepath = ARTIFACTS / 'masks.npy'):
        self.masks = np.load(filepath)
        self.masks = torch.from_numpy(self.masks).to(self.device)
        self.N = self.masks.shape[0]
        self.p = self.masks.mean().item()

    def forward(self, x, method="rise"):
        """
        x: Tensor (1, C, H, W)
        method: "rise" | "banzhaf-approx"
        returns: saliency map (num_classes, H, W)
        """
        assert method in {"rise", "banzhaf-approx"}

        N = self.N
        _, C, H, W = x.shape

        # Apply masks
        stack = self.masks * x    # (N, C, H, W)

        # Run model in batches
        outputs = []
        with torch.no_grad():
            for i in tqdm(range(0, N, self.gpu_batch), desc='Running model'):
                batch = stack[i:min(i + self.gpu_batch, N)]
                outputs.append(self.model(batch))

        outputs = torch.cat(outputs, dim=0)  # (N, num_classes)
        outputs_T = outputs.T                # (num_classes, N)

        M = self.masks.view(N, H * W).float()  # (N, HW)

        if method == "rise":
            sal = outputs_T @ M               # (num_classes, HW)
            sal = sal / (N * self.p)
        elif method == "banzhaf-approx":
            eps = 1e-6
            count_1 = M.sum(dim=0)            # (HW,)
            count_0 = N - count_1

            sum_1 = outputs_T @ M
            sum_0 = outputs_T @ (1 - M)

            sal = sum_1 / (count_1 + eps) - sum_0 / (count_0 + eps)

        n_classes = outputs.shape[1]

        sal = sal.view(n_classes, H, W)

        return sal
    