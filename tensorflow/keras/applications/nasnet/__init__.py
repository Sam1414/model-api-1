# This file is MACHINE GENERATED! Do not edit.
# Generated by: tensorflow/python/tools/api/generator/create_python_api.py script.
"""NASNet-A models for Keras.

NASNet refers to Neural Architecture Search Network, a family of models
that were designed automatically by learning the model architectures
directly on the dataset of interest.

Here we consider NASNet-A, the highest performance model that was found
for the CIFAR-10 dataset, and then extended to ImageNet 2012 dataset,
obtaining state of the art performance on CIFAR-10 and ImageNet 2012.
Only the NASNet-A models, and their respective weights, which are suited
for ImageNet 2012 are provided.

The below table describes the performance on ImageNet 2012:
--------------------------------------------------------------------------------
      Architecture       | Top-1 Acc | Top-5 Acc |  Multiply-Adds |  Params (M)
--------------------------------------------------------------------------------
|   NASNet-A (4 @ 1056)  |   74.0 %  |   91.6 %  |       564 M    |     5.3    |
|   NASNet-A (6 @ 4032)  |   82.7 %  |   96.2 %  |      23.8 B    |    88.9    |
--------------------------------------------------------------------------------

References:
  - [Learning Transferable Architectures for Scalable Image Recognition]
    (https://arxiv.org/abs/1707.07012) (CVPR 2018)

"""

from __future__ import print_function as _print_function

import sys as _sys

from tensorflow.python.keras.applications.nasnet import NASNetLarge
from tensorflow.python.keras.applications.nasnet import NASNetMobile
from tensorflow.python.keras.applications.nasnet import decode_predictions
from tensorflow.python.keras.applications.nasnet import preprocess_input

del _print_function