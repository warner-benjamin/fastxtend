# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/ffcv.operations.ipynb.

# %% ../../nbs/ffcv.operations.ipynb 1
# Contains code from:
# FFCV - Apache License 2.0 - Copyright (c) 2022 FFCV Team

# %% ../../nbs/ffcv.operations.ipynb 3
from __future__ import annotations

from abc import ABCMeta
from typing import Callable, Optional, Tuple
from dataclasses import replace

import torch
import numpy as np

from fastcore.dispatch import retain_meta
from fastcore.transform import _TfmMeta

from fastai.data.transforms import IntToFloatTensor as _IntToFloatTensor

from ffcv.pipeline.allocation_query import AllocationQuery
from ffcv.pipeline.operation import Operation
from ffcv.pipeline.state import State
from ffcv.transforms.ops import ToDevice as _ToDevice
from ffcv.transforms.ops import Convert, View

from ..imports import *

# %% auto 0
__all__ = ['ToDevice', 'ToTensorBase', 'ToTensorImage', 'ToTensorImageBW', 'ToTensorMask', 'ToTensorCategory',
           'ToTensorMultiCategory', 'ToTensorTitledTensorScalar', 'Convert', 'View']

# %% ../../nbs/ffcv.operations.ipynb 4
_all_ = ['Convert', 'View']

# %% ../../nbs/ffcv.operations.ipynb 10
class ToDevice(_ToDevice):
    "Move tensor to device and retains metadata"
    def __init__(self,
        device:int|str|torch.device, # Device to move Tensor to
        non_blocking:bool=True # Asynchronous if copying from CPU to GPU
    ):
        device, *_ = torch._C._nn._parse_to(device=device)
        super().__init__(device, non_blocking)

    def generate_code(self) -> Callable:
        def to_device(inp, dst):
            if len(inp.shape) == 4:
                if inp.is_contiguous(memory_format=torch.channels_last):
                    dst = dst.reshape(inp.shape[0], inp.shape[2], inp.shape[3], inp.shape[1])
                    dst = dst.permute(0,3,1,2)
            if not isinstance(dst, type(inp)):
                dst = retain_meta(dst, torch.as_subclass(dst, type(inp)))
            dst = dst[:inp.shape[0]]
            dst.copy_(inp, non_blocking=self.non_blocking)
            return dst
        return to_device

# %% ../../nbs/ffcv.operations.ipynb 12
class ToTensorBase(Operation):
    "Convert from Numpy array to fastai TensorBase or `tensor_cls`."
    def __init__(self, tensor_cls:TensorBase=TensorBase):
        super().__init__()
        self.tensor_cls = tensor_cls

    def generate_code(self) -> Callable:
        tensor_cls = self.tensor_cls
        def to_tensor(inp, dst):
            return tensor_cls(torch.from_numpy(inp))
        return to_tensor

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        new_dtype = torch.from_numpy(np.empty((), dtype=previous_state.dtype)).dtype
        return replace(previous_state, jit_mode=False, dtype=new_dtype), None

# %% ../../nbs/ffcv.operations.ipynb 13
class ToTensorImage(ToTensorBase):
    "Convenience op to convert from Numpy array to fastai TensorImage or `tensor_cls`."
    def __init__(self, tensor_cls:TensorImageBase=TensorImage):
        super().__init__()
        self.tensor_cls = tensor_cls

    def generate_code(self) -> Callable:
        tensor_cls = self.tensor_cls
        def to_tensor(inp, dst):
            return tensor_cls(torch.from_numpy(inp).permute(0,3,1,2))
        return to_tensor

# %% ../../nbs/ffcv.operations.ipynb 14
class ToTensorImageBW(ToTensorImage):
    "Convenience op to convert from Numpy array to fastai TensorImageBW."
    def __init__(self):
        super().__init__(TensorImageBW)

# %% ../../nbs/ffcv.operations.ipynb 15
class ToTensorMask(ToTensorImage):
    "Convenience op to convert from Numpy array to fastai TensorMask."
    def __init__(self):
        super().__init__(TensorMask)

# %% ../../nbs/ffcv.operations.ipynb 16
class ToTensorCategory(ToTensorBase):
    "Convenience op to convert from Numpy array to fastxtend TensorCategory."
    def __init__(self):
        super().__init__(TensorCategory)

# %% ../../nbs/ffcv.operations.ipynb 17
class ToTensorMultiCategory(ToTensorBase):
    "Convenience op convert from Numpy array to fastxtend TensorMultiCategory."
    def __init__(self):
        super().__init__(TensorMultiCategory)

# %% ../../nbs/ffcv.operations.ipynb 18
class ToTensorTitledTensorScalar(ToTensorBase):
    "Convenience op convert from Numpy array to fastai TitledTensorScalar."
    def __init__(self):
        super().__init__(TitledTensorScalar)