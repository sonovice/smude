__author__ = "Simon Waloschek"

import logging
import os
import argparse

import cv2 as cv
import numpy as np
import requests
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from skimage.color import gray2rgb
from skimage.io import imread, imsave
from tqdm import tqdm

from .binarize import binarize
from .model import load_model
from .mrcdi import mrcdi
from .roi import extract_roi_mask, get_border


class Smude():
    def __init__(self, use_gpu: bool = False, binarize_output: bool = True):
        """
        Instantiate new Smude object for sheet music dewarping.

        Parameters
        ----------
        use_gpu : bool, optional
            Flag if GPU should be used, by default False.
        binarize_output : bool, optional
            Flag whether the output should be binarized, by default True.
        checkpoint_path : str, optional
            Path to a trained U-Net model, by default the included 'model.ckpt'.
        """

        super().__init__()
        self.use_gpu = use_gpu
        self.binarize_output = binarize_output

        # Load Deep Learning model
        dirname = os.path.dirname(__file__)
        checkpoint_path = os.path.join(dirname, 'model.ckpt')
        if not os.path.exists(checkpoint_path):
            print('First run. Downloading model...')
            url = 'https://github.com/sonovice/smude/releases/download/v0.1.0/model.ckpt'
            response = requests.get(url, stream=True, allow_redirects=True)
            total_size_in_bytes= int(response.headers.get('content-length', 0))
            block_size = 1024 #1 Kibibyte
            progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
            with open(checkpoint_path, 'wb') as file:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)
            progress_bar.close()
            if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                print("Error: Model could not be downloaded.")
                exit(1)

        self.model = load_model(checkpoint_path)
        if self.use_gpu:
            self.model = self.model.cuda()
        self.model.freeze()

        # Define transformations on input image
        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(900),
            transforms.Grayscale(3),
            transforms.ToTensor()
        ])


    def process(self, image: np.ndarray, optimize_f: bool = False) -> np.ndarray:
        """
        Extract region of interest from sheet music image and dewarp it.

        Parameters
        ----------
        image : np.ndarray
            Input sheet music image.
        use_gpu : bool
            Flag whether to use GPU/CUDA to speed up the process.

        Returns
        -------
        np.ndarray
            Dewarped sheet music image.
        """

        if len(image.shape) < 3:
            image = gray2rgb(image)

        logging.info('Extracting ROI...')
        roi_mask, mask_ratio = extract_roi_mask(image)

        # Repeat mask for each RGB channel
        mask_3c = np.broadcast_to(roi_mask[..., None], roi_mask.shape + (3,))
        # Obtain masked result image
        result = image * mask_3c

        logging.info('Binarizing...')
        # Binarize ROI
        binarized = binarize(result)

        # Remove borders
        x_start, x_end, y_start, y_end = get_border(binarized)
        binarized = binarized[x_start:x_end, y_start:y_end]

        # Add 5% width border
        pad_width = int(binarized.shape[0] * 0.05)
        binarized = np.pad(binarized, pad_width=pad_width, mode='constant', constant_values=1)

        binarized_torch = torch.from_numpy(binarized).float()

        # Resize and convert binary image to grayscale torch tensor
        grayscale = self.transforms(binarized_torch).float()

        logging.info('Extracting features...')

        # Move to GPU
        if self.use_gpu:
            grayscale = grayscale.cuda()

        # Run inference
        output = self.model(grayscale.unsqueeze(0)).cpu()
        classes = torch.argmax(F.softmax(output[0], dim=0), dim=0)

        # Convert images to correct data types
        grayscale = (grayscale.cpu().numpy().transpose([1, 2, 0]) * 255).astype(np.uint8)
        background = (1.0 * (classes == 0).numpy() * 255).astype(np.uint8)
        upper = (1.0 * (classes == 1).numpy() * 255).astype(np.uint8)
        lower = (1.0 * (classes == 2).numpy() * 255).astype(np.uint8)
        barlines = (1.0 * (classes == 3).numpy() * 255).astype(np.uint8)
        binarized = (binarized * 255).astype(np.uint8)

        logging.info('Dewarping...')

        # Dewarp output
        cols, rows = mrcdi(
            input_img = grayscale,
            barlines_img = barlines,
            upper_img = upper,
            lower_img = lower,
            background_img = background,
            original_img = binarized,
            optimize_f = optimize_f
        )
        
        if self.binarize_output:
            dewarped = cv.remap(binarized, cols, rows, cv.INTER_CUBIC, None, cv.BORDER_CONSTANT, 255)
            # Remove border
            x_start, x_end, y_start, y_end = get_border(dewarped)
            dewarped = dewarped[x_start:x_end, y_start:y_end]

            # Add 5% min(width, height) border
            smaller = min(*dewarped.shape)
            dewarped = np.pad(dewarped, pad_width=int(smaller * 0.05), mode='constant', constant_values=255)
        else:
            # TODO rework the image manipulation part here
            # Remove borders
            image = image[x_start:x_end, y_start:y_end]
            dewarped = []
            # Do stuff for each channel individually
            for c in range(image.shape[2]):
                # Add border
                channel = np.pad(image[:, :, c], pad_width=pad_width, mode='constant', constant_values=255)
                # Dewarp
                channel = cv.remap(channel, cols, rows, cv.INTER_CUBIC, None, cv.BORDER_CONSTANT, 255)
                # Remove border again
                channel = channel[pad_width:-pad_width, pad_width:-pad_width]
                
                border_cols, border_rows = np.where(channel < 255)
                x_start = np.min(border_cols)
                x_end = np.max(border_cols) + 1
                y_start = np.min(border_rows)
                y_end = np.max(border_rows) + 1
                                
                channel = channel[x_start:x_end, y_start:y_end]
                
                dewarped.append(channel)
            dewarped = np.stack(dewarped, axis=2)
            
        return dewarped

def main():
    parser = argparse.ArgumentParser(description='Dewarp and binarize sheet music images.')
    parser.add_argument('infile', help='Specify the input image')
    parser.add_argument('-o', '--outfile', help='Specify the output image (default: result.png)', default='result.png')
    parser.add_argument('--no-binarization', help='Deactivate binarization', action='store_false')
    parser.add_argument('--use-gpu', help='use GPU', action='store_true')
    args = parser.parse_args()

    smude = Smude(use_gpu=args.use_gpu, binarize_output=args.no_binarization)

    image = imread(args.infile)
    result = smude.process(image)
    imsave(args.outfile, result)
