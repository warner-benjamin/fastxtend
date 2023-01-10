# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/callback.channelslast.ipynb.

# %% ../../nbs/callback.channelslast.ipynb 3
from __future__ import annotations

from torch.cuda.amp import GradScaler

from fastai.learner import Learner
from fastai.callback.core import Callback
from fastai.callback.fp16 import MixedPrecision

from ..imports import *

# %% auto 0
__all__ = ['ChannelsLast']

# %% ../../nbs/callback.channelslast.ipynb 5
class ChannelsLast(Callback):
    "Channels last training using PyTorch's Channels Last Memory Format (beta)"
    order = -1 # Needs to run before any model modification callbacks occur (Distributed, EMA, etc)

    def before_fit(self):
        self.learn.model.to(memory_format=torch.channels_last)

# %% ../../nbs/callback.channelslast.ipynb 8
@patch
@delegates(GradScaler)
def to_channelslast(self:Learner,
    to_fp16:bool=True, # Add `MixedPrecision` callback. Required for full channels last performance
    **kwargs
):
    "Set `Learner` and inputs to `channels_last` format and Mixed Precision by default"
    if to_fp16 and not hasattr(self, 'mixed_precision') and not hasattr(self, 'channels_last'):
        return self.add_cbs([ChannelsLast(), MixedPrecision(**kwargs)])
    elif not hasattr(self, 'channels_last'):
        return self.add_cb(ChannelsLast())

# %% ../../nbs/callback.channelslast.ipynb 9
@patch
def to_contiguous(self:Learner, to_fp32=False):
    "Set `Learner` and inputs to `contiguous_format` (default format), optionally to single precision"
    self.model.to(memory_format=torch.contiguous_format)
    if to_fp32:
        return self.remove_cbs([ChannelsLast, MixedPrecision])
    else:
        return self.remove_cb(ChannelsLast)
