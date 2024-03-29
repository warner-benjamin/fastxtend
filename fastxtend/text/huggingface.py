# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/text.huggingface.ipynb.

# %% ../../nbs/text.huggingface.ipynb 1
# Contains code from:
# blurr - Apache License 2.0 - Copyright (c) Wayde Gilliam

# %% ../../nbs/text.huggingface.ipynb 2
from __future__ import annotations

import inspect, warnings
from typing import Dict, Iterable, Sequence

import torch._dynamo as dynamo
from torch.utils.data import Sampler, Dataset
from torch.utils.data import DataLoader as _DataLoader
from torch.utils.data.dataloader import _worker_init_fn_t, _collate_fn_t

from transformers import PreTrainedModel
from transformers.modeling_utils import ModuleUtilsMixin
from transformers import logging as hf_logging

from fastai.callback.core import Callback
from fastai.losses import BaseLoss

from ..imports import *

# %% auto 0
__all__ = ['HuggingFaceLoss', 'HuggingFaceWrapper', 'HuggingFaceCallback', 'HuggingFaceLoader']

# %% ../../nbs/text.huggingface.ipynb 5
warnings.simplefilter("ignore")
hf_logging.set_verbosity_error()

# %% ../../nbs/text.huggingface.ipynb 6
class HuggingFaceLoss(BaseLoss):
    "To use the Hugging Face model's built in loss function, pass this loss to `Learner`"
    def __init__(self, **kwargs):
        self.func = None

    def __call__(self, inp:Tensor, targ:Tensor|None=None, **kwargs):
        return tensor(0.0)

# %% ../../nbs/text.huggingface.ipynb 7
class HuggingFaceWrapper(nn.Module, ModuleUtilsMixin):
    "A minimal compatibility wrapper between a Hugging Face model and `Learner`"
    def __init__(
        self,
        model: PreTrainedModel, # Hugging Face compatible model
    ):
        super().__init__()
        self.hf_model = model
        self._forward_args = inspect.getfullargspec(self.hf_model.forward).args

    def forward(self, x:Dict):
        return self.hf_model(**{k:v for k,v in x.items() if k in self._forward_args})

# %% ../../nbs/text.huggingface.ipynb 9
class HuggingFaceCallback(Callback):
    "Provides compatibility between fastai's `Learner`, the Transformers model, & `HuggingFaceLoader`"
    run_valid = True
    def __init__(self,
        labels:str|None='labels', # Input batch labels key. Set to None if input doesn't contain labels
        loss:str='loss', # Model output loss key
        logits:str='logits', # Model output logits key
        unwrap:bool=False, # After training completes, unwrap the Transformers model
    ):
        self._label_key, self._loss_key = labels, loss
        self._logit_key, self.unwrap = logits, unwrap

    def after_create(self):
        self._model_loss = isinstance(self.learn.loss_func, HuggingFaceLoss)
        if not isinstance(self.model, HuggingFaceWrapper) and not isinstance(self.model, dynamo.OptimizedModule):
            self.learn.model = HuggingFaceWrapper(self.learn.model)
            self.learn.hf_model = self.learn.model.hf_model

    def before_batch(self):
        self._loss = None
        if self._label_key is not None:
            if not self._model_loss:
                self.learn.yb = (self.xb[0].pop(self._label_key),)
            else:
                self.learn.yb = (self.xb[0][self._label_key],)
        else:
            self.learn.yb = (1,)

    def after_pred(self):
        outputs = self.learn.pred
        if self._model_loss:
            self._loss = to_float(outputs[self._loss_key])
        self.learn.pred = outputs.get(self._logit_key, None)

    def after_loss(self):
        if self._model_loss:
            self.learn.loss_grad = self._loss
            self.learn.loss = self.learn.loss_grad.clone()
        else:
            self.xb[0][self._label_key] = self.learn.yb[0]

    def after_fit(self):
        if self.unwrap:
            if isinstance(self.learn.model, dynamo.OptimizedModule) and hasattr(self.learn, 'compiler'):
                self.learn.compiler._reset_compiled()
            if isinstance(self.model, HuggingFaceWrapper):
                self.learn.model = self.learn.model.hf_model

# %% ../../nbs/text.huggingface.ipynb 11
class HuggingFaceLoader(_DataLoader):
    "A minimal compatibility DataLoader between a Hugging Face and `Learner`"
    def __init__(self,
        dataset:Dataset, # dataset from which to load the data
        batch_size:int, # Batch size
        shuffle:bool|None = None, # Randomize the order of data at each epoch (default: False)
        sampler:Sampler|Iterable|None = None, # Determines how to draw samples from the dataset. Cannot be used with shuffle.
        batch_sampler:Sampler[Sequence]|Iterable[Sequence]|None = None, # Rreturns a batch of indices at a time. Cannot be used with batch_size, shuffle, sampler, or drop_last.
        num_workers:int=0, # Number of processes to use for data loading. 0 means using the main process (default: 0).
        collate_fn:_collate_fn_t|None = None, # Function that merges a list of samples into a mini-batch of Tensors. Used for map-style datasets.
        pin_memory:bool=False, # Copy Tensors into device/CUDA pinned memory before returning them
        drop_last:bool=False, # Drop the last incomplete batch if the dataset size is not divisible by the batch size
        timeout:float=0, # Timeout value for collecting a batch from workers
        worker_init_fn:_worker_init_fn_t|None = None, # called on each worker subprocess with the worker id as input
        multiprocessing_context=None,
        generator=None,
        prefetch_factor:int|None=None, # number of batches loaded in advance by each worker
        persistent_workers:bool=False, # if True, the data loader will not shutdown the worker processes after a dataset has been consumed once
        pin_memory_device:str= "", # the data loader will copy Tensors into device pinned memory before returning them if pin_memory is set to true
    ):
        super().__init__(
            dataset=dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
            batch_sampler=batch_sampler, num_workers=num_workers, collate_fn=collate_fn,
            pin_memory=pin_memory, drop_last=drop_last, timeout=timeout,
            worker_init_fn=worker_init_fn, multiprocessing_context=multiprocessing_context,
            generator=generator, prefetch_factor=prefetch_factor, persistent_workers=persistent_workers,
            pin_memory_device=pin_memory_device
        )

    @property
    def bs(self) -> int:
        "Number of items a batch"
        return self.batch_size

    def __iter__(self):
        for b in super().__iter__():
            yield (b,)
