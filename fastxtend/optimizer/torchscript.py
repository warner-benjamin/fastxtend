# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/optimizer.torchscript.ipynb.

# %% ../../nbs/optimizer.torchscript.ipynb 2
from __future__ import annotations
from typing import Optional, Dict

from torch.nn import Parameter

from fastcore.basics import range_of, merge

from fastai.optimizer import (Optimizer, _update, weight_decay, l2_reg, average_grad, sgd_step, 
                              momentum_step, average_sqr_grad, rms_prop_step, step_stat, adam_step, 
                              radam_step, qhadam_step, larc_layer_lr, larc_step, lamb_step, Lookahead)

from ..imports import *

# %% auto 0
__all__ = ['JitOptimizer', 'SGD', 'RMSProp', 'Adam', 'RAdam', 'QHAdam', 'Larc', 'Lamb', 'JitLookahead', 'ranger']

# %% ../../nbs/optimizer.torchscript.ipynb 8
def _update(
    state:dict,
    new=None # New values to update `state` dict
):
    if isinstance(new, dict): state.update(new)

# %% ../../nbs/optimizer.torchscript.ipynb 9
class JitOptimizer(Optimizer):
    "An `Optimizer` with a modified step for TorchScript optimizers"
    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None: raise NotImplementedError("fastai optimizers currently do not support closure")
        for pg, hyper in zip(self.param_lists, self.hypers):
            for p in pg:
                if hasattr(p, 'grad') and p.grad is not None:
                    _update(self.state[p], self.cbs[0](p, p.grad, **{**self.state[p], **hyper}))

# %% ../../nbs/optimizer.torchscript.ipynb 12
@torch.jit.script
def sgd_jit_step(p:Tensor, grad:Tensor, lr:float, wd:float, mom:float, decouple_wd:bool, grad_avg:Optional[Tensor]=None, 
                 do_wd:bool=True, dampening:bool=False, force_train:Optional[bool]=None):
    dp = p
    if do_wd and wd != 0:
        if decouple_wd:
            # weight_decay
            dp = dp.mul(1 - lr*wd)
        else:
            # l2_reg
            grad = grad.add(dp, alpha=wd)

    if mom != 0:
        if grad_avg is None: 
            grad_avg = torch.zeros_like(dp, memory_format=torch.preserve_format)

        # average_grad
        damp = 1-mom if dampening else 1.
        grad_avg = grad_avg.mul(mom)
        grad_avg = grad_avg.add(grad, alpha=damp)

        # momentum_step
        dp = dp.add(grad_avg, alpha=-lr)
        p.set_(dp)
        return {'grad_avg': grad_avg}
    else:
        # sgd_step
        dp = dp.add(grad, alpha=-lr)
        p.set_(dp)
        return None

# %% ../../nbs/optimizer.torchscript.ipynb 13
def SGD(params, lr, mom=0., wd=0., decouple_wd=True, jit=False):
    "A `Optimizer` or `JitOptimizer` for SGD with `lr` and `mom` and `params`"
    if jit:
        cb = partial(sgd_jit_step, decouple_wd=decouple_wd)
        return JitOptimizer(params, cb, lr=lr, mom=mom, wd=wd)
    else:
        cbs = [weight_decay] if decouple_wd else [l2_reg]
        if mom != 0: cbs.append(average_grad)
        cbs.append(sgd_step if mom==0 else momentum_step)
        return Optimizer(params, cbs, lr=lr, mom=mom, wd=wd)

# %% ../../nbs/optimizer.torchscript.ipynb 17
@torch.jit.script
def rmsprop_jit_step(p:Tensor, grad:Tensor, lr:float, wd:float, mom:float, sqr_mom:float, eps:float, decouple_wd:bool, 
                     grad_avg:Optional[Tensor]=None, sqr_avg:Optional[Tensor]=None, do_wd:bool=True, force_train:Optional[bool]=None):
    dp = p
    if do_wd and wd != 0:
        if decouple_wd:
            # weight_decay
            dp = dp.mul(1 - lr*wd)
        else:
            # l2_reg
            grad = grad.add(dp, alpha=wd)

    if sqr_avg is None: 
        sqr_avg = torch.zeros_like(dp, memory_format=torch.preserve_format)

    if mom != 0:
        if grad_avg is None: 
            grad_avg = torch.zeros_like(dp, memory_format=torch.preserve_format)

        # average_grad, dampening=False
        grad_avg = torch.mul(grad_avg, mom)
        grad_avg = torch.add(grad_avg, grad)

        # average_sqr_grad
        sqr_avg = torch.mul(sqr_avg, sqr_mom)
        sqr_avg = torch.addcmul(sqr_avg, grad, grad, value=1-sqr_mom)

        # rms_prop_step
        denom = torch.sqrt(sqr_avg)
        denom = torch.add(denom, eps)
        dp = torch.addcdiv(dp, grad_avg, denom, value=-lr)
        p.set_(dp)
        return {'grad_avg': grad_avg, 'sqr_avg': sqr_avg}
    else:
        # average_sqr_grad
        sqr_avg = torch.mul(sqr_avg, sqr_mom)
        sqr_avg = torch.addcmul(sqr_avg, grad, grad, value=1-sqr_mom)
        
        # rms_prop_step
        denom = torch.sqrt(sqr_avg)
        denom = torch.add(denom, eps)
        dp = dp.addcdiv(grad, denom, value=-lr)
        p.set_(dp)
        return {'sqr_avg': sqr_avg}

# %% ../../nbs/optimizer.torchscript.ipynb 18
def RMSProp(params, lr, sqr_mom=0.99, mom=0., eps=1e-8, wd=0., decouple_wd=True, jit=False):
    "A `Optimizer` or `JitOptimizer` for RMSProp with `lr`, `sqr_mom`, `mom` and `params`"
    if jit:
        cb = partial(rmsprop_jit_step, decouple_wd=decouple_wd)
        return JitOptimizer(params, cb, lr=lr, mom=mom, sqr_mom=sqr_mom, eps=eps, wd=wd)
    else:
        cbs = [weight_decay] if decouple_wd else [l2_reg]
        cbs += ([average_sqr_grad] if mom==0. else [average_grad, average_sqr_grad])
        cbs.append(rms_prop_step)
        return Optimizer(params, cbs, lr=lr, mom=mom, sqr_mom=sqr_mom, wd=wd, eps=eps)

# %% ../../nbs/optimizer.torchscript.ipynb 23
def debias(beta:float, step:int):
    "Simple debias calculation"
    return 1-beta**step

# %% ../../nbs/optimizer.torchscript.ipynb 24
@torch.jit.script
def adam_jit_step(p:Tensor, grad:Tensor, lr:float, wd:float, mom:float, sqr_mom:float, eps:float, 
                  decouple_wd:bool, grad_avg:Optional[Tensor]=None, sqr_avg:Optional[Tensor]=None, 
                  do_wd:bool=True, step:int=0, force_train:Optional[bool]=None):
    dp = p
    step += 1
    if do_wd and wd != 0:
        if decouple_wd:
            # weight_decay
            dp = dp.mul(1 - lr*wd)
        else:
            # l2_reg
            grad = grad.add(dp, alpha=wd)

    if grad_avg is None: 
        grad_avg = torch.zeros_like(dp, memory_format=torch.preserve_format)
    if sqr_avg is None: 
        sqr_avg  = torch.zeros_like(dp, memory_format=torch.preserve_format)

    # average_grad, dampening=True
    grad_avg = torch.mul(grad_avg, mom)
    grad_avg = torch.add(grad_avg, grad, alpha=1-mom)

    # average_sqr_grad
    sqr_avg = torch.mul(sqr_avg, sqr_mom)
    sqr_avg = torch.addcmul(sqr_avg, grad, grad, value=1-sqr_mom)

    # adam_step
    debias1 = debias(mom, step)
    debias2 = debias(sqr_mom, step)
    dp = torch.addcdiv(dp, grad_avg, torch.sqrt(sqr_avg/debias2) + eps, value = -lr / debias1)
    p.set_(dp)

    return torch.jit.annotate(Dict[str, Union[Tensor, int]], {'grad_avg': grad_avg, 'sqr_avg': sqr_avg, 'step': step})

# %% ../../nbs/optimizer.torchscript.ipynb 25
def Adam(params, lr, mom=0.9, sqr_mom=0.99, eps=1e-5, wd=0.01, decouple_wd=True, jit=False):
    "A `Optimizer` or `JitOptimizer` for Adam with `lr`, `mom`, `sqr_mom`, `eps` and `params`"
    if jit:
        cb = partial(adam_jit_step, decouple_wd=decouple_wd)
        return JitOptimizer(params, cb, lr=lr, mom=mom, sqr_mom=sqr_mom, eps=eps, wd=wd)
    else:
        cbs = [weight_decay] if decouple_wd else [l2_reg]
        cbs += [partial(average_grad, dampening=True), average_sqr_grad, step_stat, adam_step]
        return Optimizer(params, cbs, lr=lr, mom=mom, sqr_mom=sqr_mom, eps=eps, wd=wd)

# %% ../../nbs/optimizer.torchscript.ipynb 29
@torch.jit.script
def radam_jit_step(p:Tensor, grad:Tensor, lr:float, wd:float, mom:float, sqr_mom:float, eps:float, beta:float,
                   decouple_wd:bool, grad_avg:Optional[Tensor]=None, sqr_avg:Optional[Tensor]=None,
                   do_wd:bool=True, step:int=0, force_train:Optional[bool]=None):
    dp = p
    step += 1
    if do_wd and wd != 0:
        if decouple_wd:
            # weight_decay
            dp = dp.mul(1 - lr*wd)
        else:
            # l2_reg
            grad = grad.add(dp, alpha=wd)

    if grad_avg is None: 
        grad_avg = torch.zeros_like(dp, memory_format=torch.preserve_format)
    if sqr_avg is None: 
        sqr_avg  = torch.zeros_like(dp, memory_format=torch.preserve_format)

    # average_grad, dampening=True
    grad_avg = torch.mul(grad_avg, mom)
    grad_avg = torch.add(grad_avg, grad, alpha=1-mom)

    # average_sqr_grad
    sqr_avg = torch.mul(sqr_avg, sqr_mom)
    sqr_avg = torch.addcmul(sqr_avg, grad, grad, value=1-sqr_mom)

    # radam_step
    debias1 = debias(mom, step)
    debias2 = debias(sqr_mom, step)
    r_inf = 2/(1-sqr_mom) - 1
    r = r_inf - 2*step*sqr_mom**step/(1-sqr_mom**step)
    
    if r > 5:
        v = math.sqrt(((r-4) * (r-2) * r_inf)/((r_inf-4)*(r_inf-2)*r))
        denom = torch.sqrt(sqr_avg/debias2)
        if eps != 0: 
            denom = denom + eps
        if beta != 0: 
            denom = F.softplus(denom, beta)
        dp = torch.addcdiv(dp, grad_avg, denom, value = -lr*v / debias1)
    else:
        dp = torch.add(dp, grad_avg, alpha=-lr / debias1)
    p.set_(dp)

    return torch.jit.annotate(Dict[str, Union[Tensor, int]], {'grad_avg': grad_avg, 'sqr_avg': sqr_avg, 'step': step})

# %% ../../nbs/optimizer.torchscript.ipynb 30
def RAdam(params, lr, mom=0.9, sqr_mom=0.99, eps=1e-5, wd=0., beta=0., decouple_wd=True, jit=False):
    "A `Optimizer` or `JitOptimizer` for RAdam with `lr`, `mom`, `sqr_mom`, `eps` and `params`"
    if jit:
        cb = partial(radam_jit_step, decouple_wd=decouple_wd)
        return JitOptimizer(params, cb, lr=lr, mom=mom, sqr_mom=sqr_mom, eps=eps, wd=wd, beta=beta)
    else:
        cbs = [weight_decay] if decouple_wd else [l2_reg]
        cbs += [partial(average_grad, dampening=True), average_sqr_grad, step_stat, radam_step]
        return Optimizer(params, cbs, lr=lr, mom=mom, sqr_mom=sqr_mom, eps=eps, wd=wd, beta=beta)

# %% ../../nbs/optimizer.torchscript.ipynb 34
@torch.jit.script
def qhadam_jit_step(p:Tensor, grad:Tensor, lr:float, wd:float, mom:float, sqr_mom:float, eps:float,
                    nu_1:float, nu_2:float, decouple_wd:bool, grad_avg:Optional[Tensor]=None, 
                    sqr_avg:Optional[Tensor]=None, do_wd:bool=True, step:int=0, force_train:Optional[bool]=None):
    dp = p
    step += 1
    if do_wd and wd != 0:
        if decouple_wd:
            # weight_decay
            dp = dp.mul(1 - lr*wd)
        else:
            # l2_reg
            grad = grad.add(dp, alpha=wd)

    if grad_avg is None: 
        grad_avg = torch.zeros_like(dp, memory_format=torch.preserve_format)
    if sqr_avg is None: 
        sqr_avg  = torch.zeros_like(dp, memory_format=torch.preserve_format)

    # average_grad, dampening=True
    grad_avg = torch.mul(grad_avg, mom)
    grad_avg = torch.add(grad_avg, grad, alpha=1-mom)

    # average_sqr_grad
    sqr_avg = torch.mul(sqr_avg, sqr_mom)
    sqr_avg = torch.addcmul(sqr_avg, grad, grad, value=1-sqr_mom)

    # qhadam_step
    debias1 = debias(mom, step)
    debias2 = debias(sqr_mom, step)
    dp = torch.addcdiv(dp, ((1-nu_1) * grad) + (nu_1 * (grad_avg / debias1)),
                       torch.sqrt(((1 - nu_2) * (grad)**2) + (nu_2 * (sqr_avg / debias2))) + eps,
                       value = -lr)
    p.set_(dp)

    return torch.jit.annotate(Dict[str, Union[Tensor, int]], {'grad_avg': grad_avg, 'sqr_avg': sqr_avg, 'step': step})

# %% ../../nbs/optimizer.torchscript.ipynb 35
def QHAdam(params, lr, mom=0.999, sqr_mom=0.999, nu_1=0.7, nu_2=1.0, eps=1e-8, wd=0., decouple_wd=True, jit=True):
    "An `Optimizer` for Adam with `lr`, `mom`, `sqr_mom`, `nus`, eps` and `params`"
    if jit:
        cb = partial(qhadam_jit_step, decouple_wd=decouple_wd)
        return JitOptimizer(params, cb, lr=lr, nu_1=nu_1, nu_2=nu_2, mom=mom, sqr_mom=sqr_mom, eps=eps, wd=wd)
    else:
        cbs = [weight_decay] if decouple_wd else [l2_reg]
        cbs += [partial(average_grad, dampening=True), average_sqr_grad, step_stat, qhadam_step]
        return Optimizer(params, cbs, lr=lr, nu_1=nu_1, nu_2=nu_2, mom=mom, sqr_mom=sqr_mom, eps=eps, wd=wd)

# %% ../../nbs/optimizer.torchscript.ipynb 39
@torch.jit.script
def larc_jit_step(p:Tensor, grad:Tensor, lr:float, wd:float, mom:float, eps:float, trust_coeff:float, decouple_wd:bool,
                  clip:bool, grad_avg:Optional[Tensor]=None, do_wd:bool=True, dampening:bool=False, force_train:Optional[bool]=None):
    dp = p
    if do_wd and wd != 0:
        if decouple_wd:
            # weight_decay
            dp = dp.mul(1 - lr*wd)
        else:
            # l2_reg
            grad = grad.add(dp, alpha=wd)

    # larc_layer_lr
    p_norm = torch.norm(dp)
    g_norm = torch.norm(grad)
    local_lr = lr*trust_coeff * (p_norm) / (g_norm + p_norm * wd + eps)
    if clip:
        lr = min(local_lr, lr)
    else:
        lr = local_lr

    if mom != 0:
        if grad_avg is None: 
            grad_avg = torch.zeros_like(dp, memory_format=torch.preserve_format)

        # average_grad, dampening=True
        grad_avg = torch.mul(grad_avg, mom)
        grad_avg = torch.add(grad_avg, grad)

        # larc_step
        dp = torch.add(dp, grad_avg, alpha=-lr)
    else:
        # larc_step
        dp = torch.add(dp, grad, alpha=-lr)

    p.set_(dp)
    return {'grad_avg': grad_avg}

# %% ../../nbs/optimizer.torchscript.ipynb 40
def Larc(params, lr, mom=0.9, clip=True, trust_coeff=0.02, eps=1e-8, wd=0., decouple_wd=True, jit=False):
    "A `Optimizer` for Adam with `lr`, `mom`, `sqr_mom`, `eps` and `params`"
    if jit:
        cb = partial(larc_jit_step, decouple_wd=decouple_wd, clip=clip)
        return JitOptimizer(params, cb, lr=lr, mom=mom, trust_coeff=trust_coeff, eps=eps, wd=wd)
    else:
        cbs = [weight_decay] if decouple_wd else [l2_reg]
        if mom!=0.: cbs.append(average_grad)
        cbs += [partial(larc_layer_lr, clip=clip), larc_step]
        return Optimizer(params, cbs, lr=lr, mom=mom, trust_coeff=trust_coeff, eps=eps, wd=wd)

# %% ../../nbs/optimizer.torchscript.ipynb 45
@torch.jit.script
def lamb_jit_step(p:Tensor, grad:Tensor, lr:float, wd:float, mom:float, sqr_mom:float, eps:float, 
                  decouple_wd:bool, grad_avg:Optional[Tensor]=None, sqr_avg:Optional[Tensor]=None, 
                  do_wd:bool=True, step:int=0, force_train:Optional[bool]=None):
    dp = p
    step += 1
    if do_wd and wd != 0:
        if decouple_wd:
            # weight_decay
            dp = dp.mul(1 - lr*wd)
        else:
            # l2_reg
            grad = grad.add(dp, alpha=wd)

    if grad_avg is None: 
        grad_avg = torch.zeros_like(dp, memory_format=torch.preserve_format)
    if sqr_avg is None: 
        sqr_avg  = torch.zeros_like(dp, memory_format=torch.preserve_format)

    # average_grad, dampening=True
    grad_avg = torch.mul(grad_avg, mom)
    grad_avg = torch.add(grad_avg, grad, alpha=1-mom)

    # average_sqr_grad
    sqr_avg = torch.mul(sqr_avg, sqr_mom)
    sqr_avg = torch.addcmul(sqr_avg, grad, grad, value=1-sqr_mom)

    # lamb_step
    debias1 = debias(mom, step)
    debias2 = debias(sqr_mom, step)
    r1 = dp.pow(2).mean().sqrt()
    lstep = (grad_avg/debias1) / ((sqr_avg/debias2).sqrt()+eps)
    r2 = lstep.pow(2).mean().sqrt()
    if r1 == 0 or r2 == 0:
        q = 1.
    else:
        q = min(r1/r2, 10.)
    dp = torch.add(dp,lstep, alpha = -lr * q)

    p.set_(dp)

    return torch.jit.annotate(Dict[str, Union[Tensor, int]], {'grad_avg': grad_avg, 'sqr_avg': sqr_avg, 'step': step})

# %% ../../nbs/optimizer.torchscript.ipynb 46
def Lamb(params, lr, mom=0.9, sqr_mom=0.99, eps=1e-5, wd=0., decouple_wd=True, jit=False):
    "A `Optimizer` for Adam with `lr`, `mom`, `sqr_mom`, `eps` and `params`"
    if jit:
        cb = partial(lamb_jit_step, decouple_wd=decouple_wd)
        return JitOptimizer(params, cb, lr=lr, mom=mom, sqr_mom=sqr_mom, eps=eps, wd=wd)
    else:
        cbs = [weight_decay] if decouple_wd else [l2_reg]
        cbs += [partial(average_grad, dampening=True), average_sqr_grad, step_stat, lamb_step]
        return Optimizer(params, cbs, lr=lr, mom=mom, sqr_mom=sqr_mom, eps=eps, wd=wd)

# %% ../../nbs/optimizer.torchscript.ipynb 50
class JitLookahead(Optimizer):
    "An `Optimizer` with a modified step for Lookahead TorchScript optimizers"
    def __init__(self, params:Tensor, cbs:list, train_bn:bool=True, **defaults):
        super().__init__(params, cbs, train_bn, **defaults)
        self._init_state()

    @torch.no_grad()
    def step(self, closure=None):
        self.count += 1
        if closure is not None: raise NotImplementedError("fastai optimizers currently do not support closure")
        for pg, hyper in zip(self.param_lists, self.hypers):
            for p in pg:
                if hasattr(p, 'grad') and p.grad is not None:
                    _update(self.state[p], self.cbs[0](p, p.grad, **{**self.state[p], **hyper}, count=self.count))

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

# %% ../../nbs/optimizer.torchscript.ipynb 53
@torch.jit.script
def ranger_jit_step(p:Tensor, grad:Tensor, lr:float, wd:float, mom:float, sqr_mom:float, eps:float, beta:float,
                    decouple_wd:bool, count:int, k:int, alpha:float, grad_avg:Optional[Tensor]=None, sqr_avg:Optional[Tensor]=None,
                    slow_p:Optional[Tensor]=None, do_wd:bool=True, step:int=0, force_train:Optional[bool]=None):
    dp = p
    step += 1
    if slow_p is None: 
        slow_p = dp.clone().detach()

    if do_wd and wd != 0:
        if decouple_wd:
            # weight_decay
            dp = dp.mul(1 - lr*wd)
        else:
            # l2_reg
            grad = grad.add(dp, alpha=wd)

    if grad_avg is None: 
        grad_avg = torch.zeros_like(dp, memory_format=torch.preserve_format)
    if sqr_avg is None: 
        sqr_avg  = torch.zeros_like(dp, memory_format=torch.preserve_format)

    # average_grad, dampening=True
    grad_avg = torch.mul(grad_avg, mom)
    grad_avg = torch.add(grad_avg, grad, alpha=1-mom)

    # average_sqr_grad
    sqr_avg = torch.mul(sqr_avg, sqr_mom)
    sqr_avg = torch.addcmul(sqr_avg, grad, grad, value=1-sqr_mom)

    # radam_step
    debias1 = debias(mom, step)
    debias2 = debias(sqr_mom, step)
    r_inf = 2/(1-sqr_mom) - 1
    r = r_inf - 2*step*sqr_mom**step/(1-sqr_mom**step)
    
    if r > 5:
        v = math.sqrt(((r-4) * (r-2) * r_inf)/((r_inf-4)*(r_inf-2)*r))
        denom = torch.sqrt(sqr_avg/debias2)
        if eps != 0: 
            denom = denom + eps
        if beta != 0: 
            denom = F.softplus(denom, beta)
        dp = torch.addcdiv(dp, grad_avg, denom, value = -lr*v / debias1)
    else:
        dp = torch.add(dp, grad_avg, alpha=-lr / debias1)

    # lookahead step
    if step == 1:
        p.set_(dp)
        return torch.jit.annotate(Dict[str, Union[Tensor, int]], {'grad_avg': grad_avg, 'sqr_avg': sqr_avg, 'step': step, 'slow_p': slow_p})
    elif count % k != 0:
        p.set_(dp)
        return torch.jit.annotate(Dict[str, Union[Tensor, int]], {'grad_avg': grad_avg, 'sqr_avg': sqr_avg, 'step': step})
    else:
        slow_p = torch.add(slow_p, torch.sub(dp, slow_p), alpha=alpha)
        p.set_(slow_p)
        return torch.jit.annotate(Dict[str, Union[Tensor, int]], {'grad_avg': grad_avg, 'sqr_avg': sqr_avg, 'step': step, 'slow_p': slow_p})

# %% ../../nbs/optimizer.torchscript.ipynb 54
def ranger(params, lr, mom=0.95, sqr_mom=0.99, eps=1e-6, wd=0.01, beta=0., k=6, alpha=0.5, decouple_wd=True, jit=False):
    "Convenience method for `Lookahead` with `RAdam`"
    if jit:
        cb = partial(ranger_jit_step, decouple_wd=decouple_wd)
        return JitLookahead(params, cb, lr=lr, mom=mom, sqr_mom=sqr_mom, eps=eps, wd=wd, beta=beta, k=k, alpha=alpha)
    else:
        return Lookahead(RAdam(params, lr=lr, mom=mom, sqr_mom=sqr_mom, eps=eps, wd=wd, beta=beta), k=k, alpha=alpha)
