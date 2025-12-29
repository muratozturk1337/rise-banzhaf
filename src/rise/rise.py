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
    mask_t = torch.tensor(mask).unsqueeze(0).unsqueeze(0)  # (1,1,s,s)
    mask_up = F.interpolate(mask_t, size=up_size.tolist(), mode='bilinear', align_corners=False)
    return mask_up.squeeze()

class RISE(nn.Module):
    def __init__(self, model, input_size, gpu_batch=100):
        super(RISE, self).__init__()
        self.model = model
        self.input_size = input_size
        self.gpu_batch = gpu_batch

    def generate_masks(self, N, s, p, savepath = ARTIFACTS / 'masks.npy'):
        cell_size = np.ceil(np.array(self.input_size) / s).astype(int)
        up_size = (s + 1) * cell_size   # +1 so we can shift grid

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

        self.masks = self.masks.reshape(-1, 1, *self.input_size)    # (N, 1, H, W) add dim for channel
        np.save(savepath, self.masks)
        self.masks = torch.from_numpy(self.masks)
        self.masks = self.masks.cuda()
        self.N = N
        self.p = p

    def load_masks(self, filepath = ARTIFACTS / 'masks.npy'):
        self.masks = np.load(filepath)
        self.masks = torch.from_numpy(self.masks).cuda()
        self.N = self.masks.shape[0]

    def forward(self, x):
        """
        x: Tensor (1, C, H, W)
        returns: saliency map (C, H, W)
        """
        N = self.N
        _, C, H, W = x.shape

        # Apply masks
        stack = self.masks * x    # (N, C, H, W)

        # Run model in batches
        outputs = []
        for i in range(0, N, self.gpu_batch):
            batch = stack[i:i + self.gpu_batch]
            outputs.append(self.model(batch))

        outputs = torch.cat(outputs, dim=0)  # (N, num_classes)

        # RISE aggregation
        sal = torch.matmul(
            outputs.T,                      # (num_classes, N)
            self.masks.view(N, H * W)       # (N, H*W)
        )

        sal = sal.view(outputs.shape[1], H, W)
        sal = sal / (N * self.p)

        return sal