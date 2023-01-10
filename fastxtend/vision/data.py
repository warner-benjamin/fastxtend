# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/vision.data.ipynb.

# %% ../../nbs/vision.data.ipynb 2
from __future__ import annotations

from fastcore.transform import DisplayedTransform

from fastai.data.block import TransformBlock
from fastai.data.transforms import IntToFloatTensor
from fastai.vision.core import PILBase, PILImage, PILMask, TensorPoint, TensorBBox, AddMaskCodes

from ..imports import *

# %% auto 0
__all__ = ['PreBatchAsItem', 'PostBatchAsItem', 'ImageCPUBlock', 'MaskCPUBlock']

# %% ../../nbs/vision.data.ipynb 6
class PreBatchAsItem(DisplayedTransform):
    "Converts Tensor from CHW to BCHW by adding a fake B dim"
    order=11
    def encodes(self, x:TensorImage|TensorMask|TensorPoint|TensorBBox):
        return x.unsqueeze(0)
    def decodes(self, x:TensorImage|TensorMask|TensorPoint|TensorBBox):
        return x.squeeze(0)

# %% ../../nbs/vision.data.ipynb 7
class PostBatchAsItem(DisplayedTransform):
    "Converts Tensor from BCHW to CHW by removing the fake B dim"
    order=98
    def encodes(self, x:TensorImage|TensorMask|TensorPoint|TensorBBox):
        return x.squeeze(0)
    def decodes(self, x:TensorImage|TensorMask|TensorPoint|TensorBBox):
        return x.unsqueeze(0)

# %% ../../nbs/vision.data.ipynb 10
def ImageCPUBlock(cls:PILBase=PILImage):
    "A `TransformBlock` for images of `cls` for running batch_tfms on CPU"
    return TransformBlock(type_tfms=cls.create,
                          item_tfms=[IntToFloatTensor, PreBatchAsItem, PostBatchAsItem])

# %% ../../nbs/vision.data.ipynb 11
def MaskCPUBlock(codes:Listy|None=None):
    "A `TransformBlock` for segmentation masks, potentially with `codes`, for running batch_tfms on CPU"
    return TransformBlock(type_tfms=PILMask.create,
                          item_tfms=[AddMaskCodes(codes=codes), IntToFloatTensor, PreBatchAsItem, PostBatchAsItem])
