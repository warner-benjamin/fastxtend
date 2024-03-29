# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/optimizer.sophia.ipynb.

# %% ../../nbs/optimizer.sophia.ipynb 1
# Sophia implementation based on the paper's code release
# https://github.com/Liuhong99/Sophia - MIT License - Copyright 2023 Hong Liu

# %% ../../nbs/optimizer.sophia.ipynb 4
from __future__ import annotations
from typing import Optional, Dict

import numpy as np

from torch.distributions import Categorical
from torch.nn import CrossEntropyLoss

from fastai.callback.core import Callback
from fastai.callback.fp16 import MixedPrecision
from fastai.losses import CrossEntropyLossFlat, LabelSmoothingCrossEntropy, LabelSmoothingCrossEntropyFlat
from fastai.optimizer import Optimizer, _update

from .foreach import ForEachOptimizer

from ..imports import *

# %% auto 0
__all__ = ['Sophia', 'sophia', 'SophiaCallback']

# %% ../../nbs/optimizer.sophia.ipynb 5
def sophia_step(p:Tensor, lr:float, eps:float, wd:float, mom:float, hess_mom:float,
                rho:float, bs:int, hessian_step:bool, grad_avg:Tensor|None=None,
                hessian:Tensor|None=None, do_wd:bool=True, **kwargs):
    "Updates Stable Adam moving averages and performs the Stable Adam step with `lr` on `p`"
    if grad_avg is None:
        grad_avg = torch.zeros_like(p, memory_format=torch.preserve_format)
        hessian  = torch.zeros_like(p, memory_format=torch.preserve_format)

    if hessian_step:
        hessian.mul_(hess_mom).addcmul_(p.grad.data, p.grad.data, value=1-hess_mom)
    else:
        if wd!=0 and do_wd:
            p.data.mul_(1-lr*wd)

        # update moving average
        grad_avg.mul_(mom).add_(p.grad.data, alpha=1-mom)

        # compute sophia update ratio
        ratio = grad_avg.abs().div(hessian.mul(rho * bs).add(eps)).clamp(None, 1)

        # sophia update step
        p.data.addcmul_(grad_avg.sign(), ratio, value=-lr)

    return {'grad_avg': grad_avg, 'hessian': hessian}

sophia_step.defaults = dict(mom=0.9, hess_mom=0.99)

# %% ../../nbs/optimizer.sophia.ipynb 6
class SophiaOptimizer(Optimizer):
    def __init__(self,
        params:Tensor|Iterable, # Model parameters
        cbs:callable|MutableSequence, # `Optimizer` step callbacks
        **defaults # Hyper parameters default values
    ):
        super().__init__(params, cbs, **defaults)
        self.update_sophia_hypers(0, False)

    def update_sophia_hypers(self, bs, hessian_step):
        self._bs = bs
        self._hessian_step = hessian_step

    def step(self, closure=None):
        if closure is not None: raise NotImplementedError("fastai optimizers currently do not support closure")
        for p,pg,state,hyper in self.all_params(with_grad=True):
            for cb in self.cbs:
                state = _update(state, cb(p, **{**state, **hyper}, bs=self._bs, hessian_step=self._hessian_step))
            self.state[p] = state

    def clear_state(self):
        super().clear_state()
        self.update_sophia_hypers(0, False)

# %% ../../nbs/optimizer.sophia.ipynb 7
def sophia_foreach_step(p:list[Tensor], g:list[Tensor], grad_avg:list[Tensor], hessian:list[Tensor],
                        do_wd:np.ndarray[Any, bool], lr:float, wd:float, mom:float, hess_mom:float,
                        eps:float, rho:float, bs:int, hessian_step:bool, **kwargs):
    if hessian_step:
        torch._foreach_mul_(hessian, scalar=hess_mom)
        torch._foreach_addcmul_(hessian, g, g, value=1-hess_mom)
    else:
        # weight_decay
        if wd != 0:
            wd = np.where(do_wd, 1-lr*wd, 1.)
            torch._foreach_mul_(p, scalars=wd.tolist())

        # update moving average
        torch._foreach_mul_(grad_avg, scalar=mom)
        torch._foreach_add_(grad_avg, g, alpha=1-mom)

        # compute sophia update ratio
        ratio = torch._foreach_abs(grad_avg)
        temp = torch._foreach_mul(hessian, scalar=rho*bs)
        torch._foreach_add_(temp, scalar=eps)
        torch._foreach_div_(ratio, temp)
        torch._foreach_clamp_max_(ratio, scalar=1)

        # sophia update step
        temp = [ga.sign() for ga in grad_avg]
        torch._foreach_addcmul_(p, temp, ratio, value=-lr)

# %% ../../nbs/optimizer.sophia.ipynb 8
class SophiaForEachOptimizer(ForEachOptimizer):
    "An `ForEachOptimizer` with a modified step for `sophia_foreach_step`"
    def __init__(self,
        params:Listified[Tensor], # Model parameters
        opt_step:Callable, # `ForEachOptimizer` optimizer step
        **defaults # Optimizer specific hyper parameters default values
    ):
        super().__init__(params, opt_step, True, **defaults)
        self.update_sophia_hypers(0, False)

    def update_sophia_hypers(self, bs, hessian_step):
        self._bs = bs
        self._hessian_step = hessian_step

    def clear_state(self):
        super().clear_state()
        self.update_sophia_hypers(0, False)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise NotImplementedError("fastai optimizers currently do not support closure")
        for pg, hyper in zip(self.param_lists, self.hypers):
            pl, gl, grad_avg, hessian, do_wd = [], [], [], [], []

            for p in pg:
                if p.grad is not None:
                    state = self.state[p]

                    if 'grad_avg' not in state:
                        state['grad_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['hessian']  = torch.zeros_like(p, memory_format=torch.preserve_format)

                    pl.append(p)
                    gl.append(p.grad)
                    grad_avg.append(state['grad_avg'])
                    hessian.append(state['hessian'])
                    do_wd.append(state.get('do_wd', True))

            self.opt_step(p=pl, g=gl, grad_avg=grad_avg, hessian=hessian,
                          do_wd=np.array(do_wd, dtype=bool), bs=self._bs,
                          hessian_step=self._hessian_step, **hyper)

# %% ../../nbs/optimizer.sophia.ipynb 9
def Sophia(
    params:Listified[Tensor], # Model parameters or parameter groups
    lr:float, # Default learning rate
    mom:float=0.965, # Gradient moving average (β1) coefficient
    hess_mom:float=0.99, # Hessian moving average (β2) coefficient
    rho:float=0.4, # Maximum update size, set higher for more agressive updates
    eps:float=1e-15, # Added for numerical stability
    wd:float=0.01, # Optional weight decay
    foreach:bool=False, # Use fused ForEach implementation
) -> SophiaOptimizer|SophiaForEachOptimizer:
    "A fastai Sophia optimizer with a fused ForEach implementation"
    if foreach:
        return SophiaForEachOptimizer(params, sophia_foreach_step, lr=lr, mom=mom,
                                      hess_mom=hess_mom, rho=rho, eps=eps, wd=wd)
    else:
        return SophiaOptimizer(params, [sophia_step], lr=lr, mom=mom,
                               hess_mom=hess_mom, rho=rho, eps=eps, wd=wd)

# %% ../../nbs/optimizer.sophia.ipynb 10
def sophia(
    mom:float=0.965, # Gradient moving average (β1) coefficient
    hess_mom:float=0.99, # Hessian moving average (β2) coefficient
    rho:float=0.4, # Maximum update size, set higher for more agressive updates
    eps:float=1e-15, # Added for numerical stability
    wd:float=0.01, # Optional weight decay
    foreach:bool=False, # Use fused ForEach implementation
) -> SophiaOptimizer|SophiaForEachOptimizer:
    "Partial function for the Sophia optimizer with a fused ForEach implementation"
    return partialler(Sophia, mom=mom, hess_mom=hess_mom, eps=eps,
                      rho=rho, wd=wd, foreach=foreach)

# %% ../../nbs/optimizer.sophia.ipynb 11
class SophiaHessian(str, Enum):
    "Hessian estimator for the Sophia optimizer for autocomplete"
    sophiag = 'sophiag'

# %% ../../nbs/optimizer.sophia.ipynb 12
class SophiaCallback(Callback):
    "Modifies the training loop for the Sophia Optimizer. Required for Sophia to run."
    order,run_valid = MixedPrecision.order+1,False
    def __init__(self,
        hessian_update:int=10, # Update Sophia's Hessian estimate every `hessian_update` Optimizer steps
        # hessian_est:str|SophiaHessian=SophiaHessian.sophiag # Sophia's Hessian estimator. Defaults to SophiaG's Gauss-Newton-Bartlett
    ):
        store_attr()

    def before_fit(self):
        if not isinstance(self.learn.opt, (SophiaOptimizer, SophiaForEachOptimizer)):
            raise ValueError("`SophiaCallback` only supports the `Sophia` optimizer")
        if not isinstance(self.learn.loss_func, (CrossEntropyLoss, CrossEntropyLossFlat,
                                                 LabelSmoothingCrossEntropy,
                                                 LabelSmoothingCrossEntropyFlat)):
            warn('Non-CrossEntropy loss detected, SophiaG assumes data is in a categorical distribution.')
        self._step_iter = 0
        self._hessian_step = False

    @torch.no_grad()
    def before_loss(self):
        if self._step_iter % self.hessian_update == self.hessian_update:
            dist = Categorical(logits=self.pred)
            self.learn.yb = dist.sample()
            self._hessian_step = True

    def before_step(self):
        self.learn.opt.update_sophia_hypers(find_bs(self.learn.yb), self._hessian_step)

    def after_step(self):
        self._step_iter += 1
        self._hessian_step = False
