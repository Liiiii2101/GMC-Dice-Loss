import os
import random
import typing
from collections import defaultdict
from copy import copy, deepcopy
from typing import Any

import matplotlib.pyplot as plt
import torchio as tio
import numpy as np
import scipy
import skfmm
import torch
from torch import Tensor
from torch.utils.data import Dataset
import SimpleITK as sitk
from torchvision.transforms.v2 import functional as v2F
from torchvision import transforms


def get_sdf(label, normalize=False):
    # Get inside of label volume
    struct = scipy.ndimage.generate_binary_structure(3, 1)
    inside = scipy.ndimage.binary_erosion(label.bool()[0], structure=struct, iterations=1)

    label = label.int()
    label = 1 - label  # Change outside 0 -> 1 & inside 1 -> 0

    label[0][inside] = -1  # Change inside 0 -> -1, but leave surface at 0
    sdf_label = torch.from_numpy(skfmm.distance(label).astype(np.float32))

    if normalize:
        sdf_label = sdf_label / torch.abs(sdf_label).max()  # Zero stays at zero

    # Inverse variant
    # sdf_label = torch.from_numpy(skfmm.distance(label).astype(np.float32))
    # sdf_label = sdf_label - sdf_label.max()  # TODO: temp inverse
    # sdf_label = abs(sdf_label)

    return sdf_label