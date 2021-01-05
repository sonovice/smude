#!/usr/bin/env python

from setuptools import setup

setup(
    name='smude',
    version='0.1.0',
    description='Sheet Music Dewarping',
    author='Simon Waloschek',
    url='https://github.com/sonovice/smude',
    packages=['smude'],

    entry_points={
        'console_scripts': [
            'smude = smude:main',
        ],
    },

    python_requires='>=3.8.5',

    install_requires=[
        'scikit-image>=0.18.1',
        'opencv-contrib-python>=4.5.1.48',
        'requests>=2.24.0',
        'pytorch-lightning>=1.1.2',
        'torchvision>=0.8.2',
    ],
)
