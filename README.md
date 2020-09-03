# Smude - Sheet Music Dewarping
### About

**Smude** is a library dedicated to dewarping/rectifying sheet music images. It is able to turn images like

<p align="center">
    <img src="https://github.com/sonovice/smude/raw/master/images/example_input.jpg" height="400px" />
</p>
<p align="center">
    into
</p>
<p align="center">
    <img src="https://github.com/sonovice/smude/raw/master/images/example_output.jpg" height="400px" style="border:1px solid black" />
</p>

The actual dewarping algorithm is loosely based on this paper:
> Meng, G. et. al. (2012):
> Metric Rectification of Curved Document Images.
> IEEE Transactions on Pattern Analysis and Machine Intelligence, 34(4), p. 707-722.

(DOI: [https://doi.org/10.1109/TPAMI.2011.151](https://doi.org/10.1109/TPAMI.2011.151))

### Quick Start
Clone this repository and make sure that all dependencies listed in `environment.yml` are installed, e.g. using conda:
```bash
$ git clone https://github.com/sonovice/smude.git
$ cd smude
$ conda env create -f environment.yml
$ conda activate smude
```

See `example.py` for a simple usage example:
```python
from skimage.io import imread, imsave
from smude import Smude

image = imread('images/input_fullsize.jpg')
smude = Smude(use_gpu=False)
result = smude.process(image)
imsave('result.png', result)
```
⚠️ **Smude** will download a ~348 MB Deep Learning model on the first run!

### Side Notes

**Smude** works best under these conditions:
- The entire page should be covered in the input image plus some extra margins.
- Make sure the sheet music page is evenly lit.
- Unsharp/defocused images may work but mostly won't.
- The dewarping algorithm assumes that the curved page shape is a [General Cylindric Surface](https://en.wikipedia.org/wiki/Cylinder#Cylindrical_surfaces). In practice, these are usually pages that are bound in a book and thus often exhibit a curvature when opened.

### License
This repository is under the "Commons Clause" License Condition v1.0 on top of GNU AGPLv3.
