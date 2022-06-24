[![Project Status: Active – The project has reached a stable, usable
state and is being actively
developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Python Version](https://img.shields.io/pypi/pyversions/cellshape-voxel.svg)](https://pypi.org/project/cellshape-voxel)
[![PyPI](https://img.shields.io/pypi/v/cellshape-voxel.svg)](https://pypi.org/project/cellshape-voxel)
[![Downloads](https://pepy.tech/badge/cellshape-voxel)](https://pepy.tech/project/cellshape-voxel)
[![Wheel](https://img.shields.io/pypi/wheel/cellshape-voxel.svg)](https://pypi.org/project/cellshape-voxel)
[![Development Status](https://img.shields.io/pypi/status/cellshape-voxel.svg)](https://github.com/Sentinal4D/cellshape-voxel)
[![Tests](https://img.shields.io/github/workflow/status/Sentinal4D/cellshape-voxel/tests)](
    https://github.com/Sentinal4D/cellshape-voxel/actions)
[![Coverage Status](https://coveralls.io/repos/github/Sentinal4D/cellshape-voxel/badge.svg?branch=master)](https://coveralls.io/github/Sentinal4D/cellshape-voxel?branch=master)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<img src="https://github.com/DeVriesMatt/cellshape-voxel/blob/main/img/cellshape_voxel.png" 
     alt="Cellshape logo by Matt De Vries">
___
Cellshape-voxel is an easy-to-use tool to analyse the shapes of cells using deep learning and, in particular, 3D convolutional neural networks. The tool provides the ability to train 3D convolutional autoencoders on 3D single cell masks as well as providing pre-trained networks for inference.


## To install
```bash
pip install cellshape-voxel
```

## Usage
### Basic usage
```python
import torch
from cellshape_voxel import VoxelAutoEncoder
from cellshape_voxel.encoders.resnet import Bottleneck

model = VoxelAutoEncoder(num_layers_encoder=3,
                         num_layers_decoder=3,
                         encoder_type="resnet",
                         input_shape=(64, 64, 64, 1),
                         filters=(32, 64, 128, 256, 512),
                         num_features=50,
                         bias=True,
                         activations=False,
                         batch_norm=True,
                         leaky=True,
                         neg_slope=0.01,
                         resnet_depth=10,
                         resnet_block_inplanes=(64, 128, 256, 512),
                         resnet_block=Bottleneck,
                         n_input_channels=1,
                         no_max_pool=True,
                         resnet_shortcut_type="B",
                         resnet_widen_factor=1.0)

volume = torch.randn(1, 64, 64, 64, 1)

recon, features = model(volume)
```

### To train a 3D resnet autoencoder on masks of cells or nuclei:
```python
import torch
from torch.utils.data import DataLoader
import cellshape_voxel as voxel


input_dir = "path/to/binary/mask/files/"
batch_size = 16
learning_rate = 0.0001
num_epochs = 1
output_dir = "path/to/save/output/"

model = voxel.AutoEncoder(
    num_layers_encoder=4,
    num_layers_decoder=4,
    input_shape=(64, 64, 64, 1),
    encoder_type="resnet",
)

dataset = voxel.VoxelDataset(
    PATH_TO_DATASET, transform=None, img_size=(300, 300, 300)
)

dataloader = voxel.DataLoader(dataset, batch_size=batch_size, shuffle=True)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate * 16 / batch_size,
    betas=(0.9, 0.999),
    weight_decay=1e-6,
)

voxel.train(model, dataloader, 1, optimizer, output_dir)
```


## Parameters

- `num_features`: int.  
The size of the latent space of the autoencoder. If you have rectangular images, make sure your image size is the maximum of the width and height
- `k`: int.  
The number of neightbours to use in the k-nearest-neighbours graph construction.
- `encoder_type`: int.  
The type of encoder: 'foldingnet' or 'dgcnn'
- `decoder_type`: int.  
The type of decoder: 'foldingnet' or 'dgcnn'


## For developers
* Fork the repository
* Clone your fork
```bash
git clone https://github.com/USERNAME/cellshape-voxel 
```
* Install an editable version (`-e`) with the development requirements (`dev`)
```bash
cd cellshape-voxel
pip install -e .[dev] 
```
* To install pre-commit hooks to ensure formatting is correct:
```bash
pre-commit install
```

* To release a new version:

Firstly, update the version with bump2version (`bump2version patch`, 
`bump2version minor` or `bump2version major`). This will increment the 
package version (to a release candidate - e.g. `0.0.1rc0`) and tag the 
commit. Push this tag to GitHub to run the deployment workflow:

```bash
git push --follow-tags
```

Once the release candidate has been tested, the release version can be created with:

```bash
bump2version release
```
