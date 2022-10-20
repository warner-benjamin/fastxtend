# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/optimizer.foreach.ipynb.

# %% ../../nbs/optimizer.foreach.ipynb 2
from __future__ import annotations
from typing import Optional, Dict

import numpy as np

from torch.nn import Parameter

from fastcore.basics import range_of, merge

from fastai.optimizer import Optimizer

from ..imports import *

# %% auto 0
__all__ = []

# %% ../../nbs/optimizer.foreach.ipynb 9
def sgd_foreach_step(p:list[Tensor], g:list[Tensor], no_wd_p:list[Tensor], no_wd_g:list[Tensor], grad_avg:list[Tensor], 
                     no_wd_grad_avg:list[Tensor], lr:float, wd:float, mom:float, decouple_wd:bool, dampening:bool=False, **kwargs):
    if len(p) > 0 and wd != 0:
        if decouple_wd:
            # weight_decay
            torch._foreach_mul_(p, 1 - lr * wd)
        else:
            # l2_reg
            torch._foreach_add_(g, p, alpha=wd)
        
    # combine wd and non-wd lists
    if len(no_wd_p) > 0:
        p += no_wd_p
        g += no_wd_g
        if mom != 0:
            grad_avg += no_wd_grad_avg

    if mom != 0:
        # average_grad
        damp = 1-mom if dampening else 1.
        torch._foreach_mul_(grad_avg, mom)
        torch._foreach_add_(grad_avg, g, alpha=damp)

        # momentum_step
        torch._foreach_add_(p, grad_avg, alpha=-lr)
    else:
        # sgd_step
        torch._foreach_add_(p, g, alpha=-lr)

# %% ../../nbs/optimizer.foreach.ipynb 10
class SGDForEachOptimizer(Optimizer):
    "An `Optimizer` with a modified step for SGD ForEach"
    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None: raise NotImplementedError("fastai optimizers currently do not support closure")
        for pg, hyper in zip(self.param_lists, self.hypers):
            do_wd_p, do_wd_g, do_wd_grad_avg, no_wd_p, no_wd_g, no_wd_grad_avg = [], [], [], [], [], []

            for p in pg:
                if hasattr(p, 'grad') and p.grad is not None:
                    state = self.state[p]

                    if 'grad_avg' not in state:
                        state['grad_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format) if hyper['mom'] != 0 else None

                    if hyper['wd'] != 0 and state.get('do_wd', True):
                        do_wd_p.append(p)
                        do_wd_g.append(p.grad)
                        do_wd_grad_avg.append(state['grad_avg'])
                    else:
                        no_wd_p.append(p)
                        no_wd_g.append(p.grad)
                        no_wd_grad_avg.append(state['grad_avg'])

            self.cbs[0](do_wd_p, do_wd_g, no_wd_p, no_wd_g, do_wd_grad_avg, no_wd_grad_avg, **hyper)

# %% ../../nbs/optimizer.foreach.ipynb 14
def adam_foreach_step(p:list[Tensor], g:list[Tensor], no_wd_p:list[Tensor], no_wd_g:list[Tensor], grad_avg:list[Tensor], 
                      no_wd_grad_avg:list[Tensor], sqr_avg:list[Tensor], no_wd_sqr_avg:list[Tensor], steps:np.ndarray[Any, float],
                      lr:float, wd:float, mom:float, sqr_mom:float, eps:float, decouple_wd:bool, **kwargs):

    if len(p) > 0 and wd != 0:
        if decouple_wd:
            # weight_decay
            torch._foreach_mul_(p, 1 - lr * wd)
        else:
            # l2_reg
            torch._foreach_add_(g, p, alpha=wd)
        
    # combine wd and non-wd lists
    if len(no_wd_p) > 0:
        p += no_wd_p
        g += no_wd_g
        grad_avg += no_wd_grad_avg
        sqr_avg += no_wd_sqr_avg

    # average_grad, dampening=True
    torch._foreach_mul_(grad_avg, mom)
    torch._foreach_add_(grad_avg, g, alpha=1-mom)

    # average_sqr_grad
    torch._foreach_mul_(sqr_avg, sqr_mom)
    torch._foreach_addcmul_(sqr_avg, g, g, value=1-sqr_mom)

    # adam_step
    debias1 = -lr / (1 - mom**steps)
    debias2 = np.sqrt(1 - sqr_mom**steps)

    sqr_avg_debias2 = torch._foreach_sqrt(sqr_avg)
    torch._foreach_div_(sqr_avg_debias2, debias2.tolist())
    torch._foreach_add_(sqr_avg_debias2, eps)

    torch._foreach_addcdiv_(p, grad_avg, sqr_avg_debias2, debias1.tolist())

# %% ../../nbs/optimizer.foreach.ipynb 15
class AdamForEachOptimizer(Optimizer):
    "An `Optimizer` with a modified step for Adam ForEach"
    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None: raise NotImplementedError("fastai optimizers currently do not support closure")
        for pg, hyper in zip(self.param_lists, self.hypers):
            do_wd_p, do_wd_g, do_wd_grad_avg, do_wd_sqr_avg, do_wd_steps = [], [], [], [], []
            no_wd_p, no_wd_g, no_wd_grad_avg, no_wd_sqr_avg, no_wd_steps = [], [], [], [], []

            for p in pg:
                if hasattr(p, 'grad') and p.grad is not None:
                    state = self.state[p]

                    if 'grad_avg' not in state:
                        state['grad_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['sqr_avg']  = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['step'] = 0

                    state['step'] += 1
                    if hyper['wd'] != 0 and state.get('do_wd', True):
                        do_wd_p.append(p)
                        do_wd_g.append(p.grad)
                        do_wd_grad_avg.append(state['grad_avg'])
                        do_wd_sqr_avg.append(state['sqr_avg'])
                        do_wd_steps.append(state['step'])
                    else:
                        no_wd_p.append(p)
                        no_wd_g.append(p.grad)
                        no_wd_grad_avg.append(state['grad_avg'])
                        no_wd_sqr_avg.append(state['sqr_avg'])
                        no_wd_steps.append(state['step'])

            steps = np.array([*do_wd_steps, *no_wd_steps], dtype=np.float32)
            self.cbs[0](do_wd_p, do_wd_g, no_wd_p, no_wd_g, do_wd_grad_avg, no_wd_grad_avg, do_wd_sqr_avg, no_wd_sqr_avg, steps, **hyper)

# %% ../../nbs/optimizer.foreach.ipynb 19
def radam_foreach_step(p:list[Tensor], g:list[Tensor], no_wd_p:list[Tensor], no_wd_g:list[Tensor], grad_avg:list[Tensor], 
                       no_wd_grad_avg:list[Tensor], sqr_avg:list[Tensor], no_wd_sqr_avg:list[Tensor], ones:list[Tensor], 
                       steps:np.ndarray[float], lr:float, wd:float, mom:float, sqr_mom:float, eps:float, decouple_wd:bool, **kwargs):

    if len(p) > 0 and wd != 0:
        if decouple_wd:
            # weight_decay
            torch._foreach_mul_(p, 1 - lr * wd)
        else:
            # l2_reg
            torch._foreach_add_(g, p, alpha=wd)
        
    # combine wd and non-wd lists
    if len(no_wd_p) > 0:
        p += no_wd_p
        g += no_wd_g
        grad_avg += no_wd_grad_avg
        sqr_avg += no_wd_sqr_avg

    # average_grad, dampening=True
    torch._foreach_mul_(grad_avg, mom)
    torch._foreach_add_(grad_avg, g, alpha=1-mom)

    # average_sqr_grad
    torch._foreach_mul_(sqr_avg, sqr_mom)
    torch._foreach_addcmul_(sqr_avg, g, g, value=1-sqr_mom)

    # radam_step
    debias1 = -lr / (1 - mom**steps)
    debias2 = np.sqrt(1 - sqr_mom**steps).tolist()
    
    r_inf = 2/(1-sqr_mom) - 1
    r = r_inf - 2*steps*sqr_mom**steps/(1-sqr_mom**steps)

    rect   = np.where(r > 5, debias1*np.emath.sqrt(((r-4) * (r-2) * r_inf)/((r_inf-4)*(r_inf-2)*r)), 0).tolist()
    unrect = np.where(r <= 5, debias1, 0).tolist()

    # rectified step
    sqrt_avg_debias2 = torch._foreach_sqrt(sqr_avg)
    torch._foreach_div_(sqrt_avg_debias2, debias2)
    torch._foreach_add_(sqrt_avg_debias2, eps)
    torch._foreach_addcdiv_(p, grad_avg, sqrt_avg_debias2, scalars=rect)

    # unrectified step. cannot scale with foreach_add, so divide by one with foreach_addcdiv
    torch._foreach_addcdiv_(p, grad_avg, ones, scalars=unrect)

# %% ../../nbs/optimizer.foreach.ipynb 20
class RAdamForEachOptimizer(Optimizer):
    "An `Optimizer` with a modified step for RAdam ForEach"
    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None: raise NotImplementedError("fastai optimizers currently do not support closure")
        for pg, hyper in zip(self.param_lists, self.hypers):
            do_wd_p, do_wd_g, do_wd_grad_avg, do_wd_sqr_avg, do_wd_steps, do_wd_ones = [], [], [], [], [], []
            no_wd_p, no_wd_g, no_wd_grad_avg, no_wd_sqr_avg, no_wd_steps, no_wd_ones = [], [], [], [], [], []

            for p in pg:
                if hasattr(p, 'grad') and p.grad is not None:
                    state = self.state[p]

                    if 'grad_avg' not in state:
                        state['grad_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['sqr_avg']  = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['ones']     = torch.ones_like(p, memory_format=torch.preserve_format)
                        state['step'] = 0

                    state['step'] += 1
                    if hyper['wd'] != 0 and state.get('do_wd', True):
                        do_wd_p.append(p)
                        do_wd_g.append(p.grad)
                        do_wd_grad_avg.append(state['grad_avg'])
                        do_wd_sqr_avg.append(state['sqr_avg'])
                        do_wd_ones.append(state['ones'])
                        do_wd_steps.append(state['step'])
                    else:
                        no_wd_p.append(p)
                        no_wd_g.append(p.grad)
                        no_wd_grad_avg.append(state['grad_avg'])
                        no_wd_sqr_avg.append(state['sqr_avg'])
                        no_wd_ones.append(state['ones'])
                        no_wd_steps.append(state['step'])

            steps = np.array([*do_wd_steps, *no_wd_steps], dtype=np.float32)
            ones = do_wd_ones + no_wd_ones
            self.cbs[0](do_wd_p, do_wd_g, no_wd_p, no_wd_g, do_wd_grad_avg, no_wd_grad_avg, do_wd_sqr_avg, no_wd_sqr_avg, ones, steps, **hyper)

# %% ../../nbs/optimizer.foreach.ipynb 24
def lamb_foreach_step(p:list[Tensor], g:list[Tensor], no_wd_p:list[Tensor], no_wd_g:list[Tensor], grad_avg:list[Tensor],
                      no_wd_grad_avg:list[Tensor], sqr_avg:list[Tensor], no_wd_sqr_avg:list[Tensor], ones:list[Tensor],
                      steps:np.ndarray[float], lr:float, wd:float, mom:float, sqr_mom:float, eps:float, decouple_wd:bool, **kwargs):

    if len(p) > 0 and wd != 0:
        if decouple_wd:
            # weight_decay
            torch._foreach_mul_(p, 1 - lr * wd)
        else:
            # l2_reg
            torch._foreach_add_(g, p, alpha=wd)
        
    # combine wd and non-wd lists
    if len(no_wd_p) > 0:
        p += no_wd_p
        g += no_wd_g
        grad_avg += no_wd_grad_avg
        sqr_avg += no_wd_sqr_avg

    # average_grad, dampening=True
    torch._foreach_mul_(grad_avg, mom)
    torch._foreach_add_(grad_avg, g, alpha=1-mom)

    # average_sqr_grad
    torch._foreach_mul_(sqr_avg, sqr_mom)
    torch._foreach_addcmul_(sqr_avg, g, g, value=1-sqr_mom)

    # lamb_step
    debias1 = 1 - mom**steps
    debias2 = np.sqrt(1 - sqr_mom**steps)

    debias2 = torch._foreach_div(torch._foreach_sqrt(sqr_avg), debias2.tolist())
    torch._foreach_add_(debias2, eps)
    lstep = torch._foreach_div(grad_avg, debias1.tolist())
    torch._foreach_div_(lstep, debias2)

    # there currently is no foreach_mean or foreach_where/if methods
    q = []
    for i in range(len(p)):
        r1 = p[i].pow(2).mean().sqrt().item()
        r2 = lstep[i].pow(2).mean().sqrt().item()
        if r1 == 0 or r2 == 0:
            q.append(-lr)
        else:
            q.append(-lr*min(r1/r2, 10.))

    # cannot scale with foreach_add, so divide by one with foreach_addcdiv
    torch._foreach_addcdiv_(p, lstep, ones, scalars=q)

# %% ../../nbs/optimizer.foreach.ipynb 25
class LambForEachOptimizer(RAdamForEachOptimizer):
    "An `Optimizer` with a modified step for Lamb ForEach"

# %% ../../nbs/optimizer.foreach.ipynb 29
def ranger_foreach_step(p:list[Tensor], g:list[Tensor], no_wd_p:list[Tensor], no_wd_g:list[Tensor], grad_avg:list[Tensor],
                        no_wd_grad_avg:list[Tensor], sqr_avg:list[Tensor], no_wd_sqr_avg:list[Tensor], ones:list[Tensor], 
                        slow_p:list[Tensor], steps:np.ndarray[float], lr:float, wd:float, mom:float, sqr_mom:float, eps:float, 
                        decouple_wd:bool, count:int, k:int, alpha:float, **kwargs):

    if len(p) > 0 and wd != 0:
        if decouple_wd:
            # weight_decay
            torch._foreach_mul_(p, 1 - lr * wd)
        else:
            # l2_reg
            torch._foreach_add_(g, p, alpha=wd)
        
    # combine wd and non-wd lists
    if len(no_wd_p) > 0:
        p += no_wd_p
        g += no_wd_g
        grad_avg += no_wd_grad_avg
        sqr_avg += no_wd_sqr_avg

    # average_grad, dampening=True
    torch._foreach_mul_(grad_avg, mom)
    torch._foreach_add_(grad_avg, g, alpha=1-mom)

    # average_sqr_grad
    torch._foreach_mul_(sqr_avg, sqr_mom)
    torch._foreach_addcmul_(sqr_avg, g, g, value=1-sqr_mom)

    # radam_step
    debias1 = -lr / (1 - mom**steps)
    debias2 = np.sqrt(1 - sqr_mom**steps).tolist()
    
    r_inf = 2/(1-sqr_mom) - 1
    r = r_inf - 2*steps*sqr_mom**steps/(1-sqr_mom**steps)

    rect   = np.where(r > 5, debias1*np.emath.sqrt(((r-4) * (r-2) * r_inf)/((r_inf-4)*(r_inf-2)*r)), 0).tolist()
    unrect = np.where(r <= 5, debias1, 0).tolist()

    # rectified step
    sqrt_avg_debias2 = torch._foreach_sqrt(sqr_avg)
    torch._foreach_div_(sqrt_avg_debias2, debias2)
    torch._foreach_add_(sqrt_avg_debias2, eps)
    torch._foreach_addcdiv_(p, grad_avg, sqrt_avg_debias2, scalars=rect)

    # unrectified step. cannot scale with foreach_add, so divide by one with foreach_addcdiv
    torch._foreach_addcdiv_(p, grad_avg, ones, scalars=unrect)

    if count % k == 0:
        torch._foreach_add_(slow_p, torch._foreach_sub(p, slow_p), alpha=alpha)
        # there currently is no foreach_set method
        p = [pi.set_(slow_pi) for pi, slow_pi in zip(p, slow_p)]

# %% ../../nbs/optimizer.foreach.ipynb 30
class RangerOptimizer(Optimizer):
    "An `Optimizer` with a modified step for Lookahead TorchScript optimizers"
    def __init__(self, params:Tensor, cbs:list, train_bn:bool=True, **defaults):
        super().__init__(params, cbs, train_bn, **defaults)
        self._init_state()

    @torch.no_grad()
    def step(self, closure=None):
        self.count += 1
        if closure is not None: raise NotImplementedError("fastai optimizers currently do not support closure")
        for pg, hyper in zip(self.param_lists, self.hypers):
            do_wd_p, do_wd_g, do_wd_grad_avg, do_wd_sqr_avg, do_wd_steps, do_wd_ones, do_wd_slow = [], [], [], [], [], [], []
            no_wd_p, no_wd_g, no_wd_grad_avg, no_wd_sqr_avg, no_wd_steps, no_wd_ones, no_wd_slow = [], [], [], [], [], [], []

            for p in pg:
                if hasattr(p, 'grad') and p.grad is not None:
                    state = self.state[p]

                    if 'grad_avg' not in state:
                        state['grad_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['sqr_avg']  = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['ones']     = torch.ones_like(p, memory_format=torch.preserve_format)
                        state['slow_p']   = p.clone().detach()
                        state['step'] = 0

                    state['step'] += 1
                    if hyper['wd'] != 0 and state.get('do_wd', True):
                        do_wd_p.append(p)
                        do_wd_g.append(p.grad)
                        do_wd_grad_avg.append(state['grad_avg'])
                        do_wd_sqr_avg.append(state['sqr_avg'])
                        do_wd_ones.append(state['ones'])
                        do_wd_slow.append(state['slow_p'])
                        do_wd_steps.append(state['step'])
                    else:
                        no_wd_p.append(p)
                        no_wd_g.append(p.grad)
                        no_wd_grad_avg.append(state['grad_avg'])
                        no_wd_sqr_avg.append(state['sqr_avg'])
                        no_wd_ones.append(state['ones'])
                        no_wd_slow.append(state['slow_p'])
                        no_wd_steps.append(state['step'])

            steps = np.array([*do_wd_steps, *no_wd_steps], dtype=np.float32)
            ones = do_wd_ones + no_wd_ones
            slow_p = do_wd_slow + no_wd_slow
            self.cbs[0](do_wd_p, do_wd_g, no_wd_p, no_wd_g, do_wd_grad_avg, no_wd_grad_avg, 
                        do_wd_sqr_avg, no_wd_sqr_avg, ones, slow_p, steps, count=self.count, **hyper)

    def clear_state(self):
        super().clear_state()
        self._init_state()

    def state_dict(self):
        state = super().state_dict()
        state.update({'count': self.count})
        return state

    def load_state_dict(self, sd):
        self.count = sd.pop('count')
        super().load_state_dict(sd)

    def _init_state(self): 
        self.count = 0
