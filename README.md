# Smude - Sheet Music Dewarping

**Smude** is a library dedicated to binarization and dewarping/rectification of sheet music images taken with smartphones:

<p align="center">
    <img src="https://github.com/sonovice/smude/raw/master/images/example_input.jpg" width="49%" />
    <img src="https://github.com/sonovice/smude/raw/master/images/example_output.jpg" width="49%" style="border:1px solid black" />
</p>

## Requirements

- Python 3.12+
- ~348 MB disk space for the Deep Learning model (downloaded on first run)

## Installation

### Using uv (recommended)

[uv](https://docs.astral.sh/uv/) is a fast Python package manager. Install it first, then:

```bash
git clone https://github.com/sonovice/smude.git
cd smude
uv sync
```

### Using pip

```bash
git clone https://github.com/sonovice/smude.git
cd smude
pip install .
```

## Usage

### Command Line

Installing the package adds a command-line interface called `smude`:

```bash
# With uv
uv run smude input.jpg -o output.png

# With pip installation
smude input.jpg -o output.png
```

**Options:**

```
usage: smude [-h] [-o OUTFILE] [--no-binarization] [--use-gpu] infile

Dewarp and binarize sheet music images.

positional arguments:
  infile                Specify the input image

options:
  -h, --help            show this help message and exit
  -o OUTFILE, --outfile OUTFILE
                        Specify the output image (default: result.png)
  --no-binarization     Deactivate binarization
  --use-gpu             Use GPU for inference
```

### Python Library

```python
from skimage.io import imread, imsave
from smude import Smude

image = imread("images/input_fullsize.jpg")
smude = Smude(use_gpu=False, binarize_output=True)
result = smude.process(image)
imsave("result.png", result)
```

> **Note:** Smude will download a ~348 MB Deep Learning model on the first run.

## How It Works

Rectification of sheet music pages is divided into several steps:

1. **ROI Extraction** - Extract the sheet music page from the smartphone image
2. **Adaptive Binarization** - Convert to binary image using Sauvola algorithm
3. **U-Net Segmentation** - Pixelwise segmentation into "upper staff line", "lower staff line", and "bar line" classes
4. **Vanishing Point Estimation** - Estimate the perspective vanishing point
5. **Spline Interpolation** - Fit splines to detected staff lines
6. **Dewarping** - Rectify the curved page geometry

The Deep Learning model was trained on thousands of public domain scores from [musescore.com](https://www.musescore.com), augmented and rendered with [Verovio](https://www.verovio.org), and artificially warped using code from [NVlabs/ocrodeg](https://github.com/NVlabs/ocrodeg).

The dewarping algorithm is based on:
> Meng, G. et al. (2012): *Metric Rectification of Curved Document Images.*
> IEEE Transactions on Pattern Analysis and Machine Intelligence, 34(4), p. 707-722.
> [DOI: 10.1109/TPAMI.2011.151](https://doi.org/10.1109/TPAMI.2011.151)

## Tips for Best Results

- **Full page coverage** - Include the entire page plus some extra margins
- **Even lighting** - Ensure the sheet music is evenly lit without shadows
- **Sharp focus** - Blurry or defocused images will likely fail
- **Book curvature** - Works best with pages that have a [cylindrical surface](https://en.wikipedia.org/wiki/Cylinder#Cylindrical_surfaces) curve (typical for bound books)

## License

This repository is under the "Commons Clause" License Condition v1.0 on top of GNU AGPLv3.
