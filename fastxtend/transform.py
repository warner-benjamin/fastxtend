# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/transform.ipynb (unless otherwise specified).


from __future__ import annotations


__all__ = ['BatchRandTransform']

# Cell
#nbdev_comment from __future__ import annotations
from torch.distributions import Bernoulli
from fastcore.transform import DisplayedTransform, _is_tuple, retain_type
from fastai.torch_core import find_bs
from .imports import *

# Cell
class BatchRandTransform(DisplayedTransform):
    "Randomly selects a subset of batch `b` to apply transform with per item probability `p` in `before_call`"
    do,supports,split_idx = True,[],0
    def __init__(self,
        p:Number=1., # Probability of applying Transform to each batch item
        before_call:Callable[[Tensor|tuple[Tensor,...],int],None]|None=None, # Batch preprocessing function
        **kwargs
    ):
        store_attr('p')
        super().__init__(**kwargs)
        self.before_call = ifnone(before_call,self.before_call)
        self.bernoulli = Bernoulli(p)

    def before_call(self,
        b:Tensor|tuple[Tensor,...], # Batch item(s)
        split_idx:int # Train (0) or valid (1) index
    ):
        "Randomly select `self.idxs` and set `self.do` based on `self.p` if not valid `split_idx`"
        self.idxs = self.bernoulli.sample((find_bs(b),)).bool() if not split_idx else torch.ones(find_bs(b)).bool()
        self.do = self.p==1. or self.idxs.shape[-1] > 0

    def __call__(self,
        b:Tensor|tuple[Tensor,...], # Batch item(s)
        split_idx:int, # Train (0) or valid (1) index
        **kwargs
    ) -> Tensor|tuple[Tensor,...]:
        "Call `super().__call__` if "
        self.before_call(b, split_idx=split_idx)
        return super().__call__(b, split_idx=split_idx, **kwargs) if self.do else b

    def _do_call(self,
        f, # Transform
        x:Tensor|tuple[Tensor,...], # Batch item(s)
        **kwargs
    ) -> Tensor|tuple[Tensor,...]:
        "Override `Transform._do_call` to apply transform `f` to `x[self.idxs]`"
        if not _is_tuple(x):
            if f is None: return x
            ret = f.returns(x) if hasattr(f,'returns') else None
            return retain_type(self._do_f(f, x, **kwargs), x, ret)
        res = tuple(self._do_call(f, x_, **kwargs) for x_ in x)
        return retain_type(res, x)

    def _do_f(self,
        f, # Transform
        x:Tensor, # Batch item
        **kwargs
    ) -> Tensor:
        "Apply transform `f` to `x[self.idxs]`"
        x[self.idxs] = f(x[self.idxs], **kwargs)
        return x