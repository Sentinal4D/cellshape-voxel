[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
<img src="https://github.com/DeVriesMatt/cellshape-voxel/blob/main/img/cellshape_voxel.png" 
     alt="Cellshape logo by Matt De Vries">
___
Cellshape-voxel is an easy-to-use tool to analyse the shapes of cells using deep learning and, in particular, 3D convolutional neural networks. The tool provides the ability to train 3D convolutional autoencoders on 3D single cell masks as well as providing pre-trained networks for inference.


### To install
```bash
pip install cellshape-voxel
```

### For developers
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