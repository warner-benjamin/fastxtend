# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/optimizer.torchscript.ipynb.

# %% ../../nbs/optimizer.torchscript.ipynb 3
from __future__ import annotations
from typing import Optional, Dict

from fastai.optimizer import Optimizer, _update, Lookahead

from ..imports import *

# %% auto 0
__all__ = []

# %% ../../nbs/optimizer.torchscript.ipynb 8
def _update(
    state:dict,
    new=None # New values to update `state` dict
):
    if isinstance(new, dict): state.update(new)

# %% ../../nbs/optimizer.torchscript.ipynb 9
class JitOptimizer(Optimizer):
    "An `Optimizer` with a modified step for TorchScript optimizers"
    def __init__(self,
        params:listified[Tensor], # Model parameters
        opt_step:Callable, # `JitOptimizer` optimizer step
        train_bn:bool=True, # Train normalization layers if parameter group is frozen
        decouple_wd:bool=False, # Use decoupled weight decay or L2 regularization, if applicable
        **defaults
    ):
        if notmax_torch('1.12'):
            warn(f'TorchScript optimizers are untested on PyTorch {torch.__verson__}, recommended to use 1.12 or newer')
        super().__init__(params, [None], train_bn, **defaults)
        self.opt_step = opt_step
        self.decouple_wd = decouple_wd

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None: raise NotImplementedError("fastai optimizers currently do not support closure")
        for pg, hyper in zip(self.param_lists, self.hypers):
            for p in pg:
                if hasattr(p, 'grad') and p.grad is not None:
                    state = self.state[p]
                    _update(state, self.opt_step(p=p, g=p.grad, decouple_wd=self.decouple_wd, **{**state, **hyper}))

# %% ../../nbs/optimizer.torchscript.ipynb 12
@torch.jit.script
def sgd_jit_step(p:Tensor, g:Tensor, decouple_wd:bool, lr:float, wd:float, mom:float, 
                 grad_avg:Optional[Tensor]=None, do_wd:bool=True, force_train:Optional[bool]=None):
    "SGD TorchScript compiled `JitOptimizer` step"
    param = p
    grad = g
    if do_wd and wd != 0:
        if decouple_wd:
            # weight_decay
            param = param.mul(1-lr*wd)
        else:
            # l2_reg
            grad = grad.add(param, alpha=wd)

    if mom != 0:
        if grad_avg is None: 
            grad_avg = torch.zeros_like(param, memory_format=torch.preserve_format)

        # average_grad, dampening=False
        grad_avg = grad_avg.mul(mom).add(grad)

        # momentum_step
        param = param.add(grad_avg, alpha=-lr)
        p.set_(param)
        g.set_(grad)
        return {'grad_avg': grad_avg}
    else:
        # sgd_step
        param = param.add(grad, alpha=-lr)
        p.set_(param)
        g.set_(grad)
        return None

# %% ../../nbs/optimizer.torchscript.ipynb 20
@torch.jit.script
def rmsprop_jit_step(p:Tensor, g:Tensor, lr:float, wd:float, mom:float, sqr_mom:float, eps:float, decouple_wd:bool, 
                     grad_avg:Optional[Tensor]=None, sqr_avg:Optional[Tensor]=None, do_wd:bool=True, force_train:Optional[bool]=None):
    "SGD TorchScript compiled `JitOptimizer` step"
    param = p
    grad = g
    if do_wd and wd != 0:
        if decouple_wd:
            # weight_decay
            param = param.mul(1 - lr*wd)
        else:
            # l2_reg
            grad = grad.add(param, alpha=wd)

    if sqr_avg is None: 
        sqr_avg = torch.zeros_like(param, memory_format=torch.preserve_format)

    if mom != 0:
        if grad_avg is None: 
            grad_avg = torch.zeros_like(param, memory_format=torch.preserve_format)

        # average_grad, dampening=False
        grad_avg = grad_avg.mul(mom).add(grad)

        # average_sqr_grad
        sqr_avg = sqr_avg.mul(sqr_mom).addcmul(grad, grad, value=1-sqr_mom)

        # rms_prop_step
        param = param.addcdiv(grad_avg, sqr_avg.sqrt().add(eps), value=-lr)
        p.set_(param)
        g.set_(grad)
        return {'grad_avg': grad_avg, 'sqr_avg': sqr_avg}
    else:
        # average_sqr_grad
        sqr_avg = sqr_avg.mul(sqr_mom).addcmul(grad, grad, value=1-sqr_mom)
        
        # rms_prop_step
        param = param.addcdiv(grad, sqr_avg.sqrt().add(eps), value=-lr)
        p.set_(param)
        g.set_(grad)
        return {'sqr_avg': sqr_avg}

# %% ../../nbs/optimizer.torchscript.ipynb 26
@torch.jit.script
def adam_jit_step(p:Tensor, g:Tensor, lr:float, wd:float, mom:float, sqr_mom:float, eps:float, 
                  decouple_wd:bool, grad_avg:Optional[Tensor]=None, sqr_avg:Optional[Tensor]=None, 
                  do_wd:bool=True, step:int=0, force_train:Optional[bool]=None):
    "Adam TorchScript compiled `JitOptimizer` step"
    param = p
    grad = g
    step += 1
    if do_wd and wd != 0:
        if decouple_wd:
            # weight_decay
            param = param.mul(1 - lr*wd)
        else:
            # l2_reg
            grad = grad.add(param, alpha=wd)

    if grad_avg is None: 
        grad_avg = torch.zeros_like(param, memory_format=torch.preserve_format)
    if sqr_avg is None: 
        sqr_avg  = torch.zeros_like(param, memory_format=torch.preserve_format)

    # average_grad, dampening=True
    grad_avg = grad_avg.mul(mom).add(grad, alpha=1-mom)

    # average_sqr_grad
    sqr_avg = sqr_avg.mul(sqr_mom).addcmul(grad, grad, value=1-sqr_mom)

    # adam_step
    debias1 = 1-mom**step
    debias2 = 1-sqr_mom**step
    param = torch.addcdiv(param, grad_avg, torch.sqrt(sqr_avg/debias2) + eps, value = -lr / debias1)
    p.set_(param)
    g.set_(grad)

    return torch.jit.annotate(Dict[str, Union[Tensor, int]], {'grad_avg': grad_avg, 'sqr_avg': sqr_avg, 'step': step})

# %% ../../nbs/optimizer.torchscript.ipynb 33
@torch.jit.script
def radam_jit_step(p:Tensor, g:Tensor, lr:float, wd:float, mom:float, sqr_mom:float, eps:float,
                   decouple_wd:bool, grad_avg:Optional[Tensor]=None, sqr_avg:Optional[Tensor]=None,
                   do_wd:bool=True, step:int=0, force_train:Optional[bool]=None):
    "RAdam TorchScript compiled `JitOptimizer` step"
    param = p
    grad = g
    step += 1
    if do_wd and wd != 0:
        if decouple_wd:
            # weight_decay
            param = param.mul(1 - lr*wd)
        else:
            # l2_reg
            grad = grad.add(param, alpha=wd)

    if grad_avg is None: 
        grad_avg = torch.zeros_like(param, memory_format=torch.preserve_format)
    if sqr_avg is None: 
        sqr_avg  = torch.zeros_like(param, memory_format=torch.preserve_format)

    # average_grad, dampening=True
    grad_avg = grad_avg.mul(mom).add(grad, alpha=1-mom)

    # average_sqr_grad
    sqr_avg = sqr_avg.mul(sqr_mom).addcmul(grad, grad, value=1-sqr_mom)

    # radam_step
    debias1 = 1-mom**step
    debias2 = 1-sqr_mom**step
    r_inf = 2/(1-sqr_mom) - 1
    r = r_inf - 2*step*sqr_mom**step/(1-sqr_mom**step)
    
    if r > 5:
        v = math.sqrt(((r-4)*(r-2)*r_inf)/((r_inf-4)*(r_inf-2)*r))
        denom = torch.sqrt(sqr_avg/debias2).add(eps)
        param = param.addcdiv(grad_avg, denom, value=-lr*v/debias1)
    else:
        param = param.add(grad_avg, alpha=-lr/debias1)
    p.set_(param)
    g.set_(grad)

    return torch.jit.annotate(Dict[str, Union[Tensor, int]], {'grad_avg': grad_avg, 'sqr_avg': sqr_avg, 'step': step})

# %% ../../nbs/optimizer.torchscript.ipynb 39
@torch.jit.script
def qhadam_jit_step(p:Tensor, g:Tensor, lr:float, wd:float, mom:float, sqr_mom:float, eps:float,
                    nu_1:float, nu_2:float, decouple_wd:bool, grad_avg:Optional[Tensor]=None, 
                    sqr_avg:Optional[Tensor]=None, do_wd:bool=True, step:int=0, force_train:Optional[bool]=None):
    "QHAdam TorchScript compiled `JitOptimizer` step"
    param = p
    grad = g
    step += 1
    if do_wd and wd != 0:
        if decouple_wd:
            # weight_decay
            param = param.mul(1 - lr*wd)
        else:
            # l2_reg
            grad = grad.add(param, alpha=wd)

    if grad_avg is None: 
        grad_avg = torch.zeros_like(param, memory_format=torch.preserve_format)
    if sqr_avg is None: 
        sqr_avg  = torch.zeros_like(param, memory_format=torch.preserve_format)

    # average_grad, dampening=True
    grad_avg = grad_avg.mul(mom).add(grad, alpha=1-mom)

    # average_sqr_grad
    sqr_avg = sqr_avg.mul(sqr_mom).addcmul(grad, grad, value=1-sqr_mom)

    # qhadam_step
    debias1 = 1-mom**step
    debias2 = 1-sqr_mom**step
    param = param.addcdiv(((1-nu_1) * grad) + (nu_1 * (grad_avg / debias1)),
                          torch.sqrt(((1 - nu_2) * (grad)**2) + (nu_2 * (sqr_avg / debias2))) + eps,
                          value = -lr)
    p.set_(param)
    g.set_(grad)

    return torch.jit.annotate(Dict[str, Union[Tensor, int]], {'grad_avg': grad_avg, 'sqr_avg': sqr_avg, 'step': step})

# %% ../../nbs/optimizer.torchscript.ipynb 45
@torch.jit.script
def larc_jit_step(p:Tensor, g:Tensor, lr:float, wd:float, mom:float, eps:float, trust_coeff:float, decouple_wd:bool,
                  clip:bool, grad_avg:Optional[Tensor]=None, do_wd:bool=True, dampening:bool=False, force_train:Optional[bool]=None):
    "LARC TorchScript compiled `JitOptimizer` step"
    param = p
    grad = g
    if do_wd and wd != 0:
        if decouple_wd:
            # weight_decay
            param = param.mul(1 - lr*wd)
        else:
            # l2_reg
            grad = grad.add(param, alpha=wd)

    # larc_layer_lr
    p_norm = torch.norm(param)
    g_norm = torch.norm(grad)
    local_lr = lr*trust_coeff * (p_norm) / (g_norm + p_norm * wd + eps)
    if clip:
        lr = min(local_lr, lr)
    else:
        lr = local_lr

    if mom != 0:
        if grad_avg is None: 
            grad_avg = torch.zeros_like(param, memory_format=torch.preserve_format)

        # average_grad, dampening=False
        grad_avg = grad_avg.mul(mom).add(grad)

        # larc_step
        param = torch.add(param, grad_avg, alpha=-lr)
    else:
        # larc_step
        param = torch.add(param, grad, alpha=-lr)

    p.set_(param)
    g.set_(grad)
    return {'grad_avg': grad_avg}

# %% ../../nbs/optimizer.torchscript.ipynb 52
@torch.jit.script
def lamb_jit_step(p:Tensor, g:Tensor, lr:float, wd:float, mom:float, sqr_mom:float, eps:float, 
                  decouple_wd:bool, grad_avg:Optional[Tensor]=None, sqr_avg:Optional[Tensor]=None, 
                  do_wd:bool=True, step:int=0, force_train:Optional[bool]=None):
    "LAMB TorchScript compiled `JitOptimizer` step"
    param = p
    grad = g
    step += 1
    if do_wd and wd != 0:
        if decouple_wd:
            # weight_decay
            param = param.mul(1 - lr*wd)
        else:
            # l2_reg
            grad = grad.add(param, alpha=wd)

    if grad_avg is None: 
        grad_avg = torch.zeros_like(param, memory_format=torch.preserve_format)
    if sqr_avg is None: 
        sqr_avg  = torch.zeros_like(param, memory_format=torch.preserve_format)

    # average_grad, dampening=True
    grad_avg = grad_avg.mul(mom).add(grad, alpha=1-mom)

    # average_sqr_grad
    sqr_avg = sqr_avg.mul(sqr_mom).addcmul(grad, grad, value=1-sqr_mom)

    # lamb_step
    debias1 = 1-mom**step
    debias2 = 1-sqr_mom**step

    r1 = param.pow(2).mean().sqrt()
    lstep = (grad_avg/debias1) / ((sqr_avg/debias2).sqrt()+eps)
    r2 = lstep.pow(2).mean().sqrt()
    
    if r1 == 0 or r2 == 0:
        param = param.add(lstep, alpha=-lr)
    else:
        q = min(r1/r2, 10.)
        param = param.add(lstep, alpha=-lr*q)

    p.set_(param)
    g.set_(grad)

    return torch.jit.annotate(Dict[str, Union[Tensor, int]], {'grad_avg': grad_avg, 'sqr_avg': sqr_avg, 'step': step})

# %% ../../nbs/optimizer.torchscript.ipynb 58
class JitLookahead(JitOptimizer):
    "An `JitOptimizer` with a modified step for Lookahead TorchScript optimizers"
    def __init__(self,
        params:listified[Tensor], # Model parameters
        opt_step:Callable, # `JitLookahead` optimizer step
        train_bn:bool=True, # Train normalization layers if parameter group is frozen
        decouple_wd:bool=False, # Use decoupled weight decay or L2 regularization, if applicable
        **defaults
    ):
        super().__init__(params, opt_step, train_bn, decouple_wd, **defaults)
        self._init_state()

    @torch.no_grad()
    def step(self, closure=None):
        self.count += 1
        if closure is not None: raise NotImplementedError("fastai optimizers currently do not support closure")
        for pg, hyper in zip(self.param_lists, self.hypers):
            for p in pg:
                if hasattr(p, 'grad') and p.grad is not None:
                    _update(self.state[p], self.opt_step(p, p.grad, decouple_wd=self.decouple_wd, **{**self.state[p], **hyper}, count=self.count))

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

# %% ../../nbs/optimizer.torchscript.ipynb 61
@torch.jit.script
def ranger_jit_step(p:Tensor, g:Tensor, lr:float, wd:float, mom:float, sqr_mom:float, eps:float, decouple_wd:bool, 
                    count:int, k:int, alpha:float, grad_avg:Optional[Tensor]=None, sqr_avg:Optional[Tensor]=None,
                    slow_p:Optional[Tensor]=None, do_wd:bool=True, step:int=0, force_train:Optional[bool]=None):
    "ranger TorchScript compiled `JitOptimizer` step"
    param = p
    grad = g
    step += 1
    if slow_p is None:
        slow_p = param.detach().clone()

    if do_wd and wd != 0:
        if decouple_wd:
            # weight_decay
            param = param.mul(1 - lr*wd)
        else:
            # l2_reg
            grad = grad.add(param, alpha=wd)

    if grad_avg is None: 
        grad_avg = torch.zeros_like(param, memory_format=torch.preserve_format)
    if sqr_avg is None: 
        sqr_avg  = torch.zeros_like(param, memory_format=torch.preserve_format)

    # average_grad, dampening=True
    grad_avg = grad_avg.mul(mom).add(grad, alpha=1-mom)

    # average_sqr_grad
    sqr_avg = sqr_avg.mul(sqr_mom).addcmul(grad, grad, value=1-sqr_mom)

    # radam_step
    debias1 = 1-mom**step
    debias2 = 1-sqr_mom**step
    r_inf = 2/(1-sqr_mom)-1
    r = r_inf - 2*step*sqr_mom**step/(1-sqr_mom**step)
    
    if r > 5:
        v = math.sqrt(((r-4)*(r-2)*r_inf)/((r_inf-4)*(r_inf-2)*r))
        denom = torch.sqrt(sqr_avg/debias2).add(eps)
        param = param.addcdiv(grad_avg, denom, value=-lr*v/debias1)
    else:
        param = param.add(grad_avg, alpha=-lr/debias1)

    # lookahead step
    if count % k != 0:
        p.set_(param)
        g.set_(grad)
    else:
        slow_p = slow_p.add(param.sub(slow_p), alpha=alpha)
        p.set_(slow_p)
        g.set_(grad)
    return torch.jit.annotate(Dict[str, Union[Tensor, int]], {'grad_avg': grad_avg, 'sqr_avg': sqr_avg, 'step': step, 'slow_p': slow_p})