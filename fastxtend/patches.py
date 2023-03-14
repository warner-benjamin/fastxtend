# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/patches.ipynb.

# %% auto 0
__all__ = []

# %% ../nbs/patches.ipynb 2
from packaging.version import parse

import fastai
from fastai.torch_core import _rebuild_from_type
from fastai.data.core import TfmdDL
from fastai.callback.training import ProgressCallback

from .imports import *

# %% ../nbs/patches.ipynb 3
_torch_version = parse(torch.__version__)
_torch_20  = parse('2.0')
_torch_20d = parse('2.0.0dev')
_torch_113 = parse('1.13')
_torch_112 = parse('1.12')

# %% ../nbs/patches.ipynb 5
# This has been upstreamed in fastai 2.7.11
if parse(fastai.__version__) < parse('2.7.11'):
    @patch
    def to(self:TfmdDL, device):
        self.device = device
        for tfm in self.after_batch.fs:
            # Check that tfm.to is callable as TabularPandas & transforms set tfm.to as an object
            if hasattr(tfm, 'to') and callable(tfm.to):
                tfm.to(device)
            else:
                for a in L(getattr(tfm, 'parameters', None)):
                    setattr(tfm, a, getattr(tfm, a).to(device))
        return self

# %% ../nbs/patches.ipynb 7
if parse(fastai.__version__) < parse('2.7.12'):
    @patch
    def clone(self:TensorBase, *, memory_format=None):
        cls = type(self)
        return self.as_subclass(Tensor).clone(memory_format=memory_format).as_subclass(cls)

    @patch
    def new_empty(self:TensorBase, size, *, dtype=None, layout=None, device=None, pin_memory=False, requires_grad=False):
        cls = type(self)
        if _torch_version < _torch_113 and layout is None:
            layout = torch.strided
        if _torch_version < _torch_112:
            return super(TensorBase, self).new_empty(size, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory, requires_grad=requires_grad)
        return self.as_subclass(Tensor).new_empty(size, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory, requires_grad=requires_grad).as_subclass(cls)

    @patch
    def new_empty(self:TensorBase, *size, dtype=None, layout=None, device=None, pin_memory=False, requires_grad=False):
        cls = type(self)
        if _torch_version < _torch_113 and layout is None:
            layout = torch.strided
        if _torch_version < _torch_112:
            return super(TensorBase, self).new_empty(*size, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory, requires_grad=requires_grad)
        return self.as_subclass(Tensor).new_empty(*size, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory, requires_grad=requires_grad).as_subclass(cls)

# %% ../nbs/patches.ipynb 11
if _torch_version >= _torch_20d and _torch_version < _torch_20:
    def _rebuild_tensor(storage, storage_offset, size, stride, dtype):
        # first construct a tensor with the correct dtype/device
        t = torch.tensor([], dtype=dtype, device=storage.untyped().device)
        return t.set_(storage.untyped(), storage_offset, size, stride)

    def _rebuild_tensor_v2(
        storage, storage_offset, size, stride, dtype, requires_grad, backward_hooks,
    ):
        tensor = _rebuild_tensor(storage, storage_offset, size, stride, dtype)
        tensor.requires_grad = requires_grad
        # NB: This line exists only for backwards compatibility; the
        # general expectation is that backward_hooks is an empty
        # OrderedDict.  See Note [Don't serialize hooks]
        tensor._backward_hooks = backward_hooks
        return tensor

    @patch
    def __reduce_ex__(self:TensorBase, proto):
        torch.utils.hooks.warn_if_has_hooks(self)
        args = (self.untyped_storage(), self.storage_offset(), tuple(self.size()), self.stride(), self.dtype)
        if self.is_quantized:
            args = args + (self.q_scale(), self.q_zero_point())
        args = args + (self.requires_grad, OrderedDict())
        f = torch._utils._rebuild_qtensor if self.is_quantized else _rebuild_tensor_v2
        return (_rebuild_from_type, (f, type(self), args, self.__dict__))

    @patch
    def after_batch(self:ProgressCallback):
        self.pbar.update(self.iter+1)
        if hasattr(self, 'smooth_loss'):
            self.pbar.comment = f'{self.smooth_loss.item():.4f}'