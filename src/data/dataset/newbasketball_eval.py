"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torchvision


class NewbasketballEvaluator(object):
    def __init__(self) -> None:
        pass

"""
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
COCO evaluator that works in distributed mode.
Mostly copy-paste from https://github.com/pytorch/vision/blob/edfd5a7/references/detection/coco_eval.py
The difference is that there is less copy-pasting from pycocotools
in the end of the file, as python3 can suppress prints with contextlib
"""
import os
import contextlib
import copy
import numpy as np
import torch

from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import pycocotools.mask as mask_util

from ...misc import dist_utils
from ...core import register

__all__ = ['CocoEvaluator',]


