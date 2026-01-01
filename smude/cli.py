"""
Command-line interface for Smude sheet music dewarping.
"""

__author__ = "Simon Waloschek"

import argparse

from skimage.io import imread, imsave

from . import Smude


def main():
    """Main entry point for the smude CLI."""
    parser = argparse.ArgumentParser(
        description="Dewarp and binarize sheet music images."
    )
    parser.add_argument("infile", help="Specify the input image")
    parser.add_argument(
        "-o",
        "--outfile",
        help="Specify the output image (default: result.png)",
        default="result.png",
    )
    parser.add_argument(
        "--no-binarization",
        help="Deactivate binarization",
        action="store_false",
        dest="binarize",
    )
    parser.add_argument("--use-gpu", help="Use GPU for inference", action="store_true")
    args = parser.parse_args()

    smude = Smude(use_gpu=args.use_gpu, binarize_output=args.binarize)

    image = imread(args.infile)
    result = smude.process(image)
    imsave(args.outfile, result)


if __name__ == "__main__":
    main()
