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
        'numpy==1.19.1',
        'torch==1.6.0',
        'pytorch-lightning==0.9.0',
        'scikit-image==0.17.2',
        'scipy==1.5.2',
        'torchvision==0.7.0',
        'typing==3.7.4.3',
        'typing_extensions==3.7.4.2',
        'tqdm==4.48.2',
        'requests==2.24.0',
        'opencv-contrib-python==4.4.0.42'
    ],
)
