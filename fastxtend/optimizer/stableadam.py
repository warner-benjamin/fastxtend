# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/optimizer.stableadam.ipynb.

# %% ../../nbs/optimizer.stableadam.ipynb 3
from __future__ import annotations
from typing import Optional, Dict

import numpy as np

from fastai.optimizer import Optimizer

from .foreach import ForEachOptimizer
from ..imports import *

# %% auto 0
__all__ = ['StableAdam', 'stableadam']

# %% ../../nbs/optimizer.stableadam.ipynb 6
# simplified version of:
#   beta = beta*(1-beta**(step-1))/(1-beta**step)
def debias(beta:float, step:int):
    "Stable Adam debias calculation"
    return (beta**step - beta)/(beta**step - 1)

# %% ../../nbs/optimizer.stableadam.ipynb 7
def stable_adam_step(p:Tensor, lr:float, eps:float, wd:float, mom:float, sqr_mom:float,
                     step:int=0, grad_avg:Tensor|None=None, sqr_avg:Tensor|None=None,
                     decouple_wd:bool=True, do_wd:bool=True, eps_t=None, **kwargs):
    "Updates Stable Adam moving averages and performs the Stable Adam step with `lr` on `p`"
    if step == 0:
        grad_avg = torch.zeros_like(p, memory_format=torch.preserve_format)
        sqr_avg  = torch.zeros_like(p, memory_format=torch.preserve_format)
        eps_t = tensor(eps, device=p.device, dtype=p.dtype)

    if wd!=0 and do_wd:
        if decouple_wd:
            # weight_decay
            p.data.mul_(1-lr*wd)
        else:
            # expiramental l2_reg. not in paper
            p.grad.data.add_(p.data, alpha=wd)

    # calculate debiased momentum (beta) terms
    step += 1
    db_mom = debias(mom, step)
    db_sqr_mom = debias(sqr_mom, step)

    # update moving averages (average_grad & average_sqr_grad)
    grad_avg.mul_(db_mom).add_(p.grad.data, alpha=1-db_mom)
    sqr_avg.mul_(db_sqr_mom).addcmul_(p.grad.data, p.grad.data, value=1-db_sqr_mom)

    # compute per tensor RMS stabilization term
    root_sqr_avg = sqr_avg.sqrt()
    rms = torch.norm(p.grad.data.div(root_sqr_avg.maximum(eps_t)), 2)

    # calculate RMS stabilized η_t
    lr = lr / max(1, rms)

    # stable adam step
    p.data.addcdiv_(grad_avg, root_sqr_avg.add(eps_t), value=-lr)

    return {'grad_avg':grad_avg, 'sqr_avg':sqr_avg, 'step':step, 'eps_t':eps_t}

stable_adam_step.defaults = dict(mom=0.9, sqr_mom=0.99)

# %% ../../nbs/optimizer.stableadam.ipynb 9
@torch.jit.script
def stable_adam_jit_substep(rms:Tensor, lr:float):
    return -lr / max(1, rms.item())

# %% ../../nbs/optimizer.stableadam.ipynb 10
def stable_adam_foreach_step(p:list[Tensor], g:list[Tensor], grad_avg:list[Tensor], sqr_avg:list[Tensor],
                             ones:list[Tensor], steps:np.ndarray[Any, int], do_wd:np.ndarray[Any, bool],
                             lr:float, wd:float, mom:float, sqr_mom:float, eps:float, decouple_wd:bool,
                             **kwargs):
    "Updates Stable Adam moving averages and performs the Stable Adam step with `lr` on `p`"
    if wd != 0:
        if decouple_wd:
            # weight_decay
            wd = np.where(do_wd, 1-lr*wd, 1.)
            torch._foreach_mul_(p, scalars=wd.tolist())
        else:
            # expiramental l2_reg. not in paper
            wd = np.where(do_wd, wd, 1.)
            torch._foreach_addcdiv_(g, p, ones, scalars=wd.tolist())
            # cannot use scalers with foreach_add & multiple tensors, so divide by one with foreach_addcdiv

    # calculate debiased momentum (beta) terms
    db_mom     = (mom**steps - mom)/(mom**steps - 1)
    db_sqr_mom = (sqr_mom**steps - sqr_mom)/(sqr_mom**steps - 1)

    # update moving average
    torch._foreach_mul_(grad_avg, scalars=db_mom.tolist())
    torch._foreach_addcdiv_(grad_avg, g, ones, scalars=(1-db_mom).tolist())

    # update squared moving average
    torch._foreach_mul_(sqr_avg, scalars=db_sqr_mom.tolist())
    torch._foreach_addcmul_(sqr_avg, g, g, scalars=(1-db_sqr_mom).tolist())

    # compute per tensor RMS stabilization term
    root_sqr_avg = torch._foreach_sqrt(sqr_avg)
    rms = torch._foreach_norm(torch._foreach_div(g, torch._foreach_maximum(root_sqr_avg, eps)), 2)

    # calculate RMS stabilized η_t
    lrs = [stable_adam_jit_substep(r, lr) for r in rms]

    torch._foreach_add_(root_sqr_avg, eps)
    torch._foreach_addcdiv_(p, grad_avg, root_sqr_avg, scalars=lrs)

# %% ../../nbs/optimizer.stableadam.ipynb 11
class StableAdamForEachOptimizer(ForEachOptimizer):
    "An `ForEachOptimizer` with a modified step for `stableadam_foreach_step`"
    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise NotImplementedError("fastai optimizers currently do not support closure")
        for pg, hyper in zip(self.param_lists, self.hypers):
            pl, gl, grad_avg, sqr_avg, ones, steps, do_wd = [], [], [], [], [], [], []

            for p in pg:
                if p.grad is not None:
                    state = self.state[p]

                    if 'step' not in state:
                        state['grad_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['sqr_avg']  = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['ones'] = torch.ones(1, dtype=p.dtype, device=p.device)
                        state['step'] = 0

                    state['step'] += 1
                    pl.append(p)
                    gl.append(p.grad)
                    grad_avg.append(state['grad_avg'])
                    sqr_avg.append(state['sqr_avg'])
                    ones.append(state['ones'])
                    steps.append(state['step'])
                    do_wd.append(state.get('do_wd', True))

            self.opt_step(p=pl, g=gl, grad_avg=grad_avg, sqr_avg=sqr_avg, ones=ones,
                          steps=np.array(steps, dtype=np.int32), do_wd=np.array(do_wd, dtype=bool),
                          decouple_wd=self.decouple_wd, **hyper)

# %% ../../nbs/optimizer.stableadam.ipynb 12
def StableAdam(
    params:Listified[Tensor], # Model parameters or parameter groups
    lr:float, # Default learning rate
    mom:float=0.9, # Gradient moving average (β1) coefficient
    sqr_mom:float=0.99, # Gradient squared moving average (β2) coefficient
    eps:float=1e-5, # Added for numerical stability
    wd:float=0.01, # Optional weight decay (true or L2)
    decouple_wd:bool=True, # Apply true weight decay (StableAdamW) or L2 regularization (StableAdam)
    foreach:bool=False, # Use fused ForEach implementation
) -> Optimizer|StableAdamForEachOptimizer:
    "A fastai StableAdam/StableAdamW optimizer with a fused ForEach implementation"
    if foreach:
        return StableAdamForEachOptimizer(params, stable_adam_foreach_step, lr=lr, mom=mom,
                                          sqr_mom=sqr_mom, eps=eps, wd=wd, decouple_wd=decouple_wd)
    else:
        return Optimizer(params, [stable_adam_step], lr=lr, mom=mom,
                         sqr_mom=sqr_mom, eps=eps, wd=wd, decouple_wd=decouple_wd)

# %% ../../nbs/optimizer.stableadam.ipynb 13
def stableadam(
    mom:float=0.9, # Gradient moving average (β1) coefficient
    sqr_mom:float=0.99, # Gradient squared moving average (β2) coefficient
    eps:float=1e-5, # Added for numerical stability
    wd:float=0.01, # Optional weight decay (true or L2)
    decouple_wd:bool=True, # Apply true weight decay (StableAdamW) or L2 regularization (StableAdam)
    foreach:bool=False, # Use fused ForEach implementation
) -> Optimizer|StableAdamForEachOptimizer:
    "Partial function for the StableAdam/StableAdamW optimizer with a fused ForEach implementation"
    return partialler(StableAdam, mom=mom, sqr_mom=sqr_mom, eps=eps, wd=wd,
                      decouple_wd=decouple_wd, foreach=foreach)
