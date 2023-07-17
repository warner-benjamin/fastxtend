# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/callback.amp.ipynb.

# %% ../../nbs/callback.amp.ipynb 1
# Contains code from:
# fastai - Apache License 2.0 - Copyright (c) 2023 fast.ai

# %% ../../nbs/callback.amp.ipynb 2
from __future__ import annotations

from torch.cuda.amp import GradScaler, autocast

from fastai.callback.core import Callback, CancelStepException
from fastai.learner import Learner
from fastai.torch_basics import ismin_torch

from ..imports import *

# %% auto 0
__all__ = ['AMPMode', 'MixedPrecision']

# %% ../../nbs/callback.amp.ipynb 6
class AMPMode(str, Enum):
    "Automatic mixed precision modes for ease of completion"
    FP16 = 'fp16'
    BF16 = 'bf16'

# %% ../../nbs/callback.amp.ipynb 7
@delegates(GradScaler)
class MixedPrecision(Callback):
    "Mixed precision training using Pytorch's Automatic Mixed Precision (AMP)"
    order = 10
    def __init__(self,
        amp_mode:str|AMPMode=AMPMode.FP16, # Mixed Precision training mode. Supports fp16 and bf16.
        **kwargs
    ):
        amp_mode = AMPMode(amp_mode)
        store_attr(names='amp_mode')
        self.kwargs = kwargs

    def before_fit(self):
        if self.amp_mode == AMPMode.BF16:
            if not ismin_torch("1.10"):
                raise ValueError("PyTorch 1.10 or greater required for bfloat16 mixed precision training.")
            if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
                raise ValueError("Unsuported GPU for bfloat16 mixed precision training.")
            dtype = torch.bfloat16
        elif self.amp_mode == AMPMode.FP16:
            dtype = torch.float16
        else:
            raise ValueError(f"Unrecognized precision: {self.amp_mode=}")

        # `autocast` dtype should not be set before PyTorch 1.10.
        self.autocast = autocast(dtype=dtype) if ismin_torch("1.10") else autocast()

        # `GradScaler` is not needed for bfloat16 as fp32 and bf16 have the same range
        self.kwargs['enabled'] = dtype == torch.float16
        self.learn.scaler,self.scales = GradScaler(**self.kwargs),L()

    def before_batch(self):
        self.autocast.__enter__()

    def after_pred(self):
        self.learn.pred = to_float(self.pred)

    def after_loss(self):
        self.autocast.__exit__(None, None, None)

    def before_backward(self):
        self.learn.loss_grad = self.scaler.scale(self.loss_grad)

    def before_step(self):
        "Use `self` as a fake optimizer. `self.skipped` will be set to True `after_step` if gradients overflow."
        self.skipped=True
        self.scaler.step(self)
        if self.skipped:
            raise CancelStepException()
        self.scales.append(self.scaler.get_scale())

    def after_step(self):
        self.learn.scaler.update()

    def after_fit(self):
        self.autocast,self.learn.scaler,self.scales = None,None,None

    @property
    def param_groups(self):
        "Pretend to be an optimizer for `GradScaler`"
        return self.opt.param_groups

    def step(self, *args, **kwargs):
        "Fake optimizer step to detect whether this batch was skipped from `GradScaler`"
        self.skipped=False

# %% ../../nbs/callback.amp.ipynb 10
@patch
@delegates(GradScaler)
def to_fp16(self:Learner, **kwargs):
    "Set `Learner` to float16 mixed precision using PyTorch AMP"
    return self.add_cb(MixedPrecision(**kwargs))

# %% ../../nbs/callback.amp.ipynb 11
@patch
def to_bf16(self:Learner):
    "Set `Learner` to bfloat16 mixed precision using PyTorch AMP"
    return self.add_cb(MixedPrecision(amp_mode=AMPMode.BF16))

# %% ../../nbs/callback.amp.ipynb 12
@patch
def to_fp32(self:Learner):
    "Set `Learner` to float32 precision"
    return self.remove_cb(MixedPrecision)
