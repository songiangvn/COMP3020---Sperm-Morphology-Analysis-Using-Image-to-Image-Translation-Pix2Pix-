# HoloStain
This repository includes supplementary material for training the generative model used in the following article: Y. N. Nygate, M. Levi, S. Mirsky, N. A. Turko, M. Rubin, I. Barnea, M. Haifler, A. Shalev, and N. T. Shaked, Holographic virtual staining of individual biological cells, Proceedings of the National Academy of Sciences of the U.S.A. (PNAS), 2020

Code and datasets are also available at: http://www.eng.tau.ac.il/~omni/HoloStain

## Requirements

- Linux or OSX
- NVIDIA GPU: Driver 390.59, CUDA 9.0, cuDNN 7.1.4
- Python 3.6.4
- TensorFlow 1.10.1 (the GPU version)
- PIL 5.2.0

## Train

Models are saved to './output/checkpoints' (can be changed by passing 'check_dir=your_dir' in main.py).

Run 'python main.py --epochs <number_of_epochs> --mode "train"'

(Run python 'main.py -h' to see all of the options)

## Test

Run 'python main.py --mode "test"'

License
----
Copyright (c) 2020, under the Attribution-NonCommercial 4.0 International License (https://creativecommons.org/licenses/by-nc/4.0/legalcode).

www.pnas.org/cgi/doi/10.1073/pnas.1919569117
