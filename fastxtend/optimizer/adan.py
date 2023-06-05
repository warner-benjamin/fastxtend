# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/optimizer.adan.ipynb.

# %% ../../nbs/optimizer.adan.ipynb 1
# Memory and operations reduction ported from the official Adan implementation
# https://github.com/sail-sg/Adan - Apache License 2.0 - Copyright 2022 Xingyu Xie et al

# %% ../../nbs/optimizer.adan.ipynb 4
from __future__ import annotations
from typing import Optional, Dict

import numpy as np

from fastai.optimizer import Optimizer

from .foreach import ForEachOptimizer
from .torchscript import JitOptimizer
from ..imports import *

# %% auto 0
__all__ = ['Adan', 'adan', 'AdanLargeBatchLR']

# %% ../../nbs/optimizer.adan.ipynb 5
def debias(beta:float, step:int):
    "Simple debias calculation"
    return 1-beta**step

# %% ../../nbs/optimizer.adan.ipynb 7
def adan_setup(p:Tensor, step:int=0, grad_avg:Tensor|None=None, diff_avg:Tensor|None=None,
               sqr_avg:Tensor|None=None, prior_grad:Tensor|None=None, paper_init:bool=False, **kwargs):
    "Handles Adan setup and keeps track of steps"
    if step == 0:
        grad_avg = torch.zeros_like(p, memory_format=torch.preserve_format)
        diff_avg = torch.zeros_like(p, memory_format=torch.preserve_format)
        sqr_avg  = torch.zeros_like(p, memory_format=torch.preserve_format)
        if paper_init:
            prior_grad = p.grad.clone().mul_(-1)
        else:
            prior_grad = torch.zeros_like(p, memory_format=torch.preserve_format)
        step += 1
        return {'grad_avg':grad_avg, 'diff_avg':diff_avg, 'sqr_avg':sqr_avg, 'prior_grad':prior_grad, 'step':step}
    else:
        step += 1
        return {'step':step}

# %% ../../nbs/optimizer.adan.ipynb 8
def adan_step(p:Tensor, lr:float, eps:float, wd:float, beta1:float, beta2:float, beta3:float,
              step:int, grad_avg:Tensor, diff_avg:Tensor, sqr_avg:Tensor, prior_grad:Tensor,
              do_wd:bool=True, **kwargs):
    "Updates Adan moving averages and performs the Adan step with `lr` on `p`"

    # difference between current and previous gradients, prior_grad is negated in last step
    prior_grad = torch.add(p.grad.data, prior_grad)

    # update m_k
    grad_avg.mul_(beta1).add_(p.grad.data, alpha=1-beta1)

    # update v_k
    diff_avg.mul_(beta2).add_(prior_grad, alpha=1-beta2)

    # update n_k
    prior_grad = torch.add(p.grad.data, prior_grad, alpha=beta2)
    sqr_avg.mul_(beta3).addcmul_(prior_grad, prior_grad, value=1-beta3)

    # calculate debias terms
    db1 = 1/debias(beta1, step)
    db2 = beta2/debias(beta2, step)
    db3 = lr*math.sqrt(debias(beta3, step))

    # calculate applied λ
    wd = (1+lr*wd) if wd!=0 and do_wd else 1

    # calculate η_k
    lrs = torch.sqrt(sqr_avg).div(db3).add(eps)

    # perform Adan step and apply to parameter `p`
    p.data.addcdiv_(grad_avg, lrs, value=-db1)
    p.data.addcdiv_(diff_avg, lrs, value=-db2)
    p.data.div_(wd)

    # set next step's prior_grad as negated current grad
    prior_grad = p.grad.data.clone().mul_(-1)
    return {'grad_avg':grad_avg, 'diff_avg':diff_avg, 'sqr_avg':sqr_avg, 'prior_grad':prior_grad}

adan_step.defaults = dict(beta1=0.98, beta2=0.92, beta3=0.99)

# %% ../../nbs/optimizer.adan.ipynb 10
@torch.jit.script
def adan_jit_step(p:Tensor, g:Tensor, lr:float, wd:float, beta1:float, beta2:float, beta3:float, eps:float,
                  paper_init:bool, grad_avg:Optional[Tensor]=None, diff_avg:Optional[Tensor]=None,
                  sqr_avg:Optional[Tensor]=None, prior_grad:Optional[Tensor]=None, do_wd:bool=True, step:int=0,
                  force_train:Optional[bool]=None, mom:Optional[float]=None, decouple_wd:bool=False):
    dp = p
    grad = g
    step += 1

    if grad_avg is None:
        grad_avg = torch.zeros_like(dp, memory_format=torch.preserve_format)
    if diff_avg is None:
        diff_avg = torch.zeros_like(dp, memory_format=torch.preserve_format)
    if sqr_avg is None:
        sqr_avg = torch.zeros_like(dp, memory_format=torch.preserve_format)
    if prior_grad is None:
        if paper_init:
            prior_grad = grad.clone()
        else:
            prior_grad = torch.zeros_like(dp, memory_format=torch.preserve_format)

    # difference between current and previous gradients
    diff_grad = grad.sub(prior_grad)

    # update m_k
    grad_avg = grad_avg.mul(beta1).add(grad, alpha=1-beta1)

    # update v_k
    diff_avg = diff_avg.mul(beta2).add(diff_grad, alpha=1-beta2)

    # update n_k
    adjusted_grad = grad.add(diff_grad, alpha=beta2)
    sqr_avg = sqr_avg.mul(beta3).addcmul(adjusted_grad, adjusted_grad, value=1-beta3)

    # calculate debias terms
    db1 = 1-beta1**step
    db2 = 1-beta2**step
    db3 = math.sqrt(1-beta3**step)

    # calculate applied λ
    if wd!=0 and do_wd:
        wd = (1+lr*wd)
    else:
        wd = 1.

    # calculate η_k
    lrs = lr/torch.sqrt(sqr_avg).div(db3).add(eps)

    # perform Adan step
    dp = dp.sub(torch.add(grad_avg.div(db1), diff_avg.div(db2), alpha=beta2).mul(lrs)).div(wd)

    # set next step's prior_grad as negated current grad
    prior_grad = grad.clone()

    # apply results to parameter p
    p.set_(dp)
    g.set_(grad)

    return torch.jit.annotate(Dict[str, Union[Tensor, int]], {'grad_avg':grad_avg, 'diff_avg':diff_avg, 'sqr_avg':sqr_avg, 'prior_grad':prior_grad, 'step':step})

# %% ../../nbs/optimizer.adan.ipynb 12
def adan_foreach_step(p:list[Tensor], grad:list[Tensor], grad_avg:list[Tensor], diff_avg:list[Tensor],
                      sqr_avg:list[Tensor], prior_grad:list[Tensor], steps:np.ndarray[Any, int],
                      do_wd:np.ndarray[Any, bool], lr:float, wd:float, beta1:float, beta2:float,
                      beta3:float, eps:float, **kwargs):

    # difference between current and previous gradients, prior_grad is negated in last step
    torch._foreach_add_(prior_grad, grad)

    # update m_k
    torch._foreach_mul_(grad_avg, beta1)
    torch._foreach_add_(grad_avg, grad, alpha=1-beta1)

    # update v_k
    torch._foreach_mul_(diff_avg, beta2)
    torch._foreach_add_(diff_avg, prior_grad, alpha=1-beta2)

    # update n_k
    torch._foreach_mul_(prior_grad, scalar=beta2)
    torch._foreach_add_(prior_grad, grad)
    torch._foreach_mul_(sqr_avg, beta3)
    torch._foreach_addcmul_(sqr_avg, prior_grad, prior_grad, value=1-beta3)

    # calculate debias terms
    db1 = -1/(1 - beta1**steps)
    db2 = -beta2/(1 - beta2**steps)
    db3 = lr*np.sqrt(1 - beta3**steps)

    # calculate η_k
    lrs = torch._foreach_sqrt(sqr_avg)
    torch._foreach_div_(lrs, scalars=db3.tolist())
    torch._foreach_add_(lrs, scalar=eps)

    # perform Adan step
    torch._foreach_addcdiv_(p, grad_avg, lrs, scalars=db1.tolist())
    torch._foreach_addcdiv_(p, diff_avg, lrs, scalars=db2.tolist())

    # calculate and apply λ
    if wd != 0:
        wd = np.where(do_wd, 1+lr*wd, 1.)
        torch._foreach_div_(p, scalars=wd.tolist())

    # set next step's prior_grad as negated current grad
    torch._foreach_zero_(prior_grad)
    torch._foreach_add_(prior_grad, grad, alpha=-1)

# %% ../../nbs/optimizer.adan.ipynb 13
class AdanForEachOptimizer(ForEachOptimizer):
    "An `Optimizer` with a modified step for Adan ForEach"
    def __init__(self,
        params:Listified[Tensor], # Model parameters
        opt_step:Callable, # `ForEachOptimizer` optimizer step
        paper_init:bool=False, # Initialize first prior_grad to grad following paper or zeros
        **defaults # Optimizer specific hyper parameters
    ):
        super().__init__(params, opt_step, **defaults)
        self.paper_init = paper_init

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None: raise NotImplementedError("fastai optimizers currently do not support closure")
        for pg, hyper in zip(self.param_lists, self.hypers):
            pl, gl, grad_avg, diff_avg, sqr_avg, prior_grad, steps, do_wd = [], [], [], [], [], [], [], []

            for p in pg:
                if p.grad is not None:
                    state = self.state[p]

                    if 'step' not in state:
                        state['grad_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['diff_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['sqr_avg']  = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if self.paper_init:
                            state['prior_grad'] = p.grad.clone().mul_(-1)
                        else:
                            state['prior_grad'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['step'] = 0

                    state['step'] += 1
                    pl.append(p)
                    gl.append(p.grad)
                    grad_avg.append(state['grad_avg'])
                    diff_avg.append(state['diff_avg'])
                    sqr_avg.append(state['sqr_avg'])
                    prior_grad.append(state['prior_grad'])
                    do_wd.append(state.get('do_wd', True))
                    steps.append(state['step'])

            self.opt_step(p=pl, grad=gl, grad_avg=grad_avg, diff_avg=diff_avg, sqr_avg=sqr_avg,
                          prior_grad=prior_grad, steps=np.array(steps, dtype=np.int32), do_wd=np.array(do_wd, dtype=bool), **hyper)

# %% ../../nbs/optimizer.adan.ipynb 15
def Adan(
    params:Listified[Tensor], # Model parameters or parameter groups
    lr:float, # Default learning rate
    beta1:float=0.98, # Gradient moving average (β1) coefficient
    beta2:float=0.92, # Gradient difference moving average (β2) coefficient
    beta3:float=0.99, # Gradient squared moving average (β3) coefficient
    eps:float=1e-8, # Added for numerical stability
    wd:float=0.02, # True weight decay
    paper_init:bool=False, # Initialize prior gradient with current gradient per paper, or zeroes
    foreach:bool=False, # Use fused ForEach implementation
    jit:bool=False # Use fused TorchScript implementation
) -> Optimizer|AdanForEachOptimizer|JitOptimizer:
    "A fastai Adan optimizer with optional ForEach and TorchScript implementations"
    if foreach:
        return AdanForEachOptimizer(params, adan_foreach_step, lr=lr, beta1=beta1, beta2=beta2,
                                    beta3=beta3, eps=eps, wd=wd, paper_init=paper_init)
    elif jit:
        cb = partial(adan_jit_step, paper_init=paper_init)
        return JitOptimizer(params, cb, lr=lr, beta1=beta1, beta2=beta2, beta3=beta3, eps=eps, wd=wd)
    else:
        cbs = [partial(adan_setup, paper_init=paper_init), adan_step]
        return Optimizer(params, cbs, lr=lr, beta1=beta1, beta2=beta2, beta3=beta3, eps=eps, wd=wd)

# %% ../../nbs/optimizer.adan.ipynb 16
def adan(
    beta1:float=0.98, # Gradient moving average (β1) coefficient
    beta2:float=0.92, # Gradient difference moving average (β2) coefficient
    beta3:float=0.99, # Gradient squared moving average (β3) coefficient
    eps:float=1e-8, # Added for numerical stability
    wd:float=0.02, # True weight decay
    paper_init:bool=False, # Initialize prior gradient with current gradient per paper, or zeroes
    foreach:bool=False, # Use fused ForEach implementation
    jit:bool=False # Use fused TorchScript implementation
) -> Optimizer|AdanForEachOptimizer|JitOptimizer:
    "Partial function for the Adan optimizer with fused ForEach and TorchScript implementations"
    return partialler(Adan, beta1=beta1, beta2=beta2, beta3=beta3, eps=eps, wd=wd,
                      paper_init=paper_init, foreach=foreach, jit=jit)

# %% ../../nbs/optimizer.adan.ipynb 18
def AdanLargeBatchLR(bs:int) -> float:
    "Square root rule for scaling `Adan` learning rate for large-batch training"
    return math.sqrt(bs/256)*6.25e-3
