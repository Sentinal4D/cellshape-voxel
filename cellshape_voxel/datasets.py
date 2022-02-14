import torch
from torch.utils.data import Dataset
from skimage import io
import numpy as np
from pathlib import Path


def pad_img(img, new_size):
    new_z, new_y, new_x = new_size[0], new_size[1], new_size[2]
    z, y, x = img.shape[0], img.shape[1], img.shape[2]
    assert (
        (new_z >= z) and (new_y >= y) and (new_x >= x)
    ), "New image dimensions must be larger or equal to old dimensions"

    delta_z = new_z - z
    delta_y = new_y - y
    delta_x = new_x - x

    if delta_z % 2 == 1:
        z_padding = (delta_z // 2, delta_z // 2 + 1)
    else:
        z_padding = (delta_z // 2, delta_z // 2)

    if delta_y % 2 == 1:
        y_padding = (delta_y // 2, delta_y // 2 + 1)
    else:
        y_padding = (delta_y // 2, delta_y // 2)

    if delta_x % 2 == 1:
        x_padding = (delta_x // 2, delta_x // 2 + 1)
    else:
        x_padding = (delta_x // 2, delta_x // 2)

    padded_data = np.pad(img, (z_padding, y_padding, x_padding), "constant")
    return padded_data


class VoxelDataset(Dataset):
    def __init__(self, img_dir, transform=None, img_size=(128, 128, 128)):

        self.img_dir = img_dir
        self.p = Path(self.img_dir)
        self.files = list(self.p.glob("**/*.tif"))
        self.transform = transform
        self.img_size = img_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # read the image
        file = self.files[idx]
        image = io.imread(str(file)).astype(np.float16)
        image = pad_img(image, self.img_size)

        if self.transform:
            image = self.transform(image)

        image = torch.tensor(image).type(torch.FloatTensor)
        if len(image.shape) < 5:
            image = image.unsqueeze(0)

        return image
