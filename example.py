from skimage.io import imread, imsave
from smude import Smude

smude = Smude(use_gpu=False, binarize_output=True)

image = imread('images/input_fullsize.jpg')
result = smude.process(image)
imsave('result.png', result)
