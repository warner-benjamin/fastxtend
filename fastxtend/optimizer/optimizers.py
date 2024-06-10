# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/optimizer.ipynb.

# %% ../../nbs/optimizer.ipynb 2
# Contains code from:
# fastai - Apache License 2.0 - Copyright (c) 2023 fast.ai

# %% ../../nbs/optimizer.ipynb 5
from __future__ import annotations

from fastcore.basics import partialler

from fastai.optimizer import (Optimizer, weight_decay, l2_reg, average_grad, average_sqr_grad,
                               step_stat, qhadam_step, larc_layer_lr, larc_step, lamb_step, rms_prop_step)

try:
    from fastxtend.optimizer.optimi import (AdamOptimiOptimizer, AdanOptimiOptimizer, LionOptimiOptimizer,
                                            RAdamOptimiOptimizer, RangerOptimiOptimizer, SGDOptimiOptimizer,
                                            StableAdamWOptimiOptimizer)
    OPTIMI = True
except ImportError:
    OPTIMI = False

try:
    from packaging.version import parse
    import bitsandbytes
    from fastxtend.optimizer.eightbit import (SGD8bitOptimizer, RMSProp8bitOptimizer, AdamW8bitOptimizer,
                                              LARS8bitOptimizer, LAMB8bitOptimizer, Lion8bitOptimizer)
    EIGHTBIT = True
except ImportError:
    EIGHTBIT = False

from ..imports import *

# %% auto 0
__all__ = ['Adam', 'adam', 'Lion', 'lion', 'SGD', 'sgd', 'Adan', 'adan', 'RAdam', 'radam', 'Ranger', 'ranger', 'StableAdamW',
           'stableadamw', 'Larc', 'larc', 'Lamb', 'lamb', 'RMSProp', 'rmsprop', 'QHAdam', 'qhadam']

# %% ../../nbs/optimizer.ipynb 9
def Adam(
    params:Listified[Tensor], # Model parameters or parameter groups
    lr:float, # Default learning rate
    mom:float=0.9, # Gradient moving average (β1) coefficient
    sqr_mom:float=0.99, # Gradient squared moving average (β2) coefficient
    eps:float=1e-5, # Added for numerical stability
    wd:float=0.01, # Optional weight decay
    decouple_wd:bool=True, # Apply decoupled weight decay (AdamW) instead of L2 penalty (Adam)
    decouple_lr:bool=False, # Apply fully decoupled weight decay (AdamW) instead of L2 penalty (Adam). Unsupported for `eightbit=True`.
    kahan_sum:bool|None=None, # More accurate parameter updates when training in low precision (float16 or bfloat16). If unspecified, automatically applies for low precision parameters. Unsupported for `eightbit=True`. (float16 or bfloat16). If unspecified, automatically applies for low precision parameters. Unsupported for `eightbit=True`.
    foreach:bool|None=None, # Use ForEach implementation. If unspecified, tries to use foreach over for-loop implementation.
    eightbit:bool=False, # Use bitsandbytes' eight-bit implementation instead of optimi's implementation.
    **eightbitargs # Additional eight-bit arguments. See `AdamW8bitOptimizer` for details.
) -> AdamOptimiOptimizer|AdamW8bitOptimizer:
    "A fastai Adam/AdamW optimizer with low precision, foreach, and eight-bit implementations"

    if not eightbit:
        if OPTIMI:
            return AdamOptimiOptimizer(params, lr=lr, mom=mom, sqr_mom=sqr_mom, wd=wd,
                                       eps=eps, decouple_wd=decouple_wd, decouple_lr=decouple_lr,
                                       kahan_sum=kahan_sum, foreach=foreach)
        else:
            raise ImportError('optimi package not found. Run `pip install torch-optimi`.')
    else:
        if EIGHTBIT:
            if (not decouple_wd and wd > 0) or (decouple_lr and wd > 0):
                raise NotImplementedError(f'Eight-bit Adam only supports decoupled weight decay: {decouple_wd=}')
            return AdamW8bitOptimizer(params, lr=lr, mom=mom, sqr_mom=sqr_mom, eps=eps, wd=wd, **eightbitargs)
        else:
            raise ImportError(f'{eightbit=}. bitsandbytes package not found. Run `pip install bitsandbytes`.')

# %% ../../nbs/optimizer.ipynb 10
def adam(
    mom:float=0.9, # Gradient moving average (β1) coefficient
    sqr_mom:float=0.99, # Gradient squared moving average (β2) coefficient
    eps:float=1e-5, # Added for numerical stability
    wd:float=0.01, # Optional weight decay
    decouple_wd:bool=True, # Apply decoupled weight decay (AdamW) instead of L2 penalty (Adam)
    decouple_lr:bool=False, # Apply fully decoupled weight decay (AdamW) instead of L2 penalty (Adam). Unsupported for `eightbit=True`.
    kahan_sum:bool|None=None, # More accurate parameter updates when training in low precision (float16 or bfloat16). If unspecified, automatically applies for low precision parameters. Unsupported for `eightbit=True`. (float16 or bfloat16). If unspecified, automatically applies for low precision parameters. Unsupported for `eightbit=True`.
    foreach:bool|None=None, # Use faster ForEach implementation. If unspecified, tries to use foreach over for-loop implementation.
    eightbit:bool=False, # Use bitsandbytes' eight-bit implementation instead of optimi's implementation.
    **eightbitargs # Additional eight-bit arguments. See `AdamW8bitOptimizer` for details.
) -> AdamOptimiOptimizer|AdamW8bitOptimizer:
    "A fastai-compatible Adam/AdamW optimizer with low precision, foreach, and eight-bit implementations"
    return partialler(Adam, mom=mom, sqr_mom=sqr_mom, eps=eps, wd=wd, decouple_wd=decouple_wd,
                      decouple_lr=decouple_lr, kahan_sum=kahan_sum, foreach=foreach,
                      eightbit=eightbit, **eightbitargs)

# %% ../../nbs/optimizer.ipynb 12
def Lion(
    params: Listified[Tensor],  # Model parameters or parameter groups
    lr: float,  # Default learning rate
    beta1: float = 0.9,  # Update gradient moving average (β1) coefficient
    beta2: float = 0.99,  # Gradient moving average (β2) coefficient
    wd: float = 0.1,  # Decoupled weight decay
    decouple_lr: bool = False,  # Apply fully decoupled weight decay
    kahan_sum: bool | None = None,  # More accurate parameter updates when training in low precision (float16 or bfloat16). If unspecified, automatically applies for low precision parameters. Unsupported for `eightbit=True`.
    foreach: bool | None = None,  # Use faster ForEach implementation. If unspecified, tries to use foreach over for-loop implementation.
    eightbit: bool = False,  # Use bitsandbytes' eight-bit implementation
    **eightbitargs  # Additional eight-bit arguments
) -> LionOptimiOptimizer | Lion8bitOptimizer:
    "A fastai-compatible Lion optimizer with low precision, foreach, and eight-bit implementations"

    if not eightbit:
        if OPTIMI:
            return LionOptimiOptimizer(params, lr=lr, beta1=beta1, beta2=beta2, wd=wd,
                                       decouple_lr=decouple_lr, kahan_sum=kahan_sum, foreach=foreach)
        else:
            raise ImportError('optimi package not found. Run `pip install torch-optimi`.')
    else:
        if EIGHTBIT:
            if decouple_lr and wd > 0:
                raise NotImplementedError('Eight-bit Lion only supports decoupled weight decay.')
            return Lion8bitOptimizer(params, lr=lr, beta1=beta1, beta2=beta2, wd=wd, **eightbitargs)
        else:
            raise ImportError(f'{eightbit=}. bitsandbytes package not found. Run `pip install bitsandbytes`.')


# %% ../../nbs/optimizer.ipynb 13
def lion(
    beta1: float = 0.9,  # Update gradient moving average (β1) coefficient
    beta2: float = 0.99,  # Gradient moving average (β2) coefficient
    wd: float = 0.1,  # Decoupled weight decay
    decouple_lr: bool = False,  # Apply fully decoupled weight decay
    kahan_sum: bool | None = None,  # More accurate parameter updates when training in low precision (float16 or bfloat16). If unspecified, automatically applies for low precision parameters. Unsupported for `eightbit=True`.
    foreach: bool | None = None,  # Use faster ForEach implementation. If unspecified, tries to use foreach over for-loop implementation.
    eightbit: bool = False,  # Use bitsandbytes' eight-bit implementation
    **eightbitargs  # Additional eight-bit arguments
) -> LionOptimiOptimizer | Lion8bitOptimizer:
    "A partial function for the Lion optimizer with low precision, foreach, and eight-bit implementations"
    return partialler(Lion, beta1=beta1, beta2=beta2, wd=wd, decouple_lr=decouple_lr,
                      kahan_sum=kahan_sum, foreach=foreach, eightbit=eightbit, **eightbitargs)


# %% ../../nbs/optimizer.ipynb 15
def SGD(
    params: Listified[Tensor],  # Model parameters or parameter groups
    lr: float,  # Default learning rate
    mom: float = 0.,  # Gradient moving average (β1) coefficient
    wd: float = 0.,  # Optional weight decay (decoupled or L2)
    decouple_wd: bool = True,  # Apply decoupled weight decay (SGDW) or L2 regularization (SGD)
    decouple_lr: bool = False,  # Apply fully decoupled weight decay
    kahan_sum: bool | None = None,  # More accurate parameter updates when training in low precision (float16 or bfloat16). If unspecified, automatically applies for low precision parameters. Unsupported for `eightbit=True`.
    foreach: bool | None = None,  # Use faster ForEach implementation. If unspecified, tries to use foreach over for-loop implementation.
    eightbit: bool = False,  # Use bitsandbytes' eight-bit implementation
    **eightbitargs  # Additional eight-bit arguments
) -> SGDOptimiOptimizer | SGD8bitOptimizer:
    "A fastai-compatible SGD optimizer with low precision, foreach, and eight-bit implementations"

    if not eightbit:
        if OPTIMI:
            return SGDOptimiOptimizer(params, lr=lr, mom=mom, wd=wd, decouple_wd=decouple_wd,
                                      decouple_lr=decouple_lr, kahan_sum=kahan_sum, foreach=foreach)
        else:
            raise ImportError('optimi package not found. Run `pip install torch-optimi`.')
    else:
        if EIGHTBIT:
            if decouple_wd and wd > 0:
                raise NotImplementedError('Eight-bit SGD only supports L2 weight decay.')
            return SGD8bitOptimizer(params, lr=lr, mom=mom, wd=wd, **eightbitargs)
        else:
            raise ImportError(f'{eightbit=}. bitsandbytes package not found. Run `pip install bitsandbytes`.')


# %% ../../nbs/optimizer.ipynb 16
def sgd(
    mom: float = 0.,  # Gradient moving average (β1) coefficient
    wd: float = 0.,  # Optional weight decay (decoupled or L2)
    decouple_wd: bool = True,  # Apply decoupled weight decay (SGDW) or L2 regularization (SGD)
    decouple_lr: bool = False,  # Apply fully decoupled weight decay
    kahan_sum: bool | None = None,  # More accurate parameter updates when training in low precision (float16 or bfloat16). If unspecified, automatically applies for low precision parameters. Unsupported for `eightbit=True`.
    foreach: bool | None = None,  # Use faster ForEach implementation. If unspecified, tries to use foreach over for-loop implementation.
    eightbit: bool = False,  # Use bitsandbytes' eight-bit implementation
    **eightbitargs  # Additional eight-bit arguments
) -> SGDOptimiOptimizer | SGD8bitOptimizer:
    "Partial function for the SGD optimizer with low precision, foreach, and eight-bit implementations"
    return partialler(SGD, mom=mom, wd=wd, decouple_wd=decouple_wd, decouple_lr=decouple_lr,
                      kahan_sum=kahan_sum, foreach=foreach, eightbit=eightbit, **eightbitargs)


# %% ../../nbs/optimizer.ipynb 20
def Adan(
    params: Listified[Tensor],  # Model parameters or parameter groups
    lr: float,  # Default learning rate
    beta1: float = 0.98,  # Gradient moving average (β1) coefficient
    beta2: float = 0.92,  # Gradient difference moving average (β2) coefficient
    beta3: float = 0.99,  # Gradient squared moving average (β3) coefficient
    eps: float = 1e-8,  # Added for numerical stability
    wd: float = 0.02,  # Decoupled weight decay
    decouple_lr: bool = False,  # Apply fully decoupled weight decay
    adam_wd: bool = False,  # Apply weight decay before parameter update (Adam-style), instead of after the update per Adan algorithm
    kahan_sum: bool | None = None,  # More accurate parameter updates when training in low precision (float16 or bfloat16). If unspecified, automatically applies for low precision parameters.
    foreach: bool | None = None,  # Use faster ForEach implementation. If unspecified, tries to use foreach over for-loop implementation.
) -> AdanOptimiOptimizer:
    "A fastai-compatible Adan optimizer with low precision and foreach implementations"
    if OPTIMI:
        return AdanOptimiOptimizer(params, lr=lr, beta1=beta1, beta2=beta2, beta3=beta3, wd=wd,
                                   eps=eps, decouple_lr=decouple_lr, adam_wd=adam_wd,
                                   kahan_sum=kahan_sum, foreach=foreach)
    else:
        raise ImportError('optimi package not found. Run `pip install torch-optimi`.')


# %% ../../nbs/optimizer.ipynb 21
def adan(
    beta1: float = 0.98,  # Gradient moving average (β1) coefficient
    beta2: float = 0.92,  # Gradient difference moving average (β2) coefficient
    beta3: float = 0.99,  # Gradient squared moving average (β3) coefficient
    eps: float = 1e-8,  # Added for numerical stability
    wd: float = 0.02,  # Decoupled weight decay
    decouple_lr: bool = False,  # Apply fully decoupled weight decay
    adam_wd: bool = False,  # Apply weight decay before parameter update (Adam-style), instead of after the update per Adan algorithm
    kahan_sum: bool | None = None,  # More accurate parameter updates when training in low precision (float16 or bfloat16). If unspecified, automatically applies for low precision parameters.
    foreach: bool | None = None,  # Use faster ForEach implementation. If unspecified, tries to use foreach over for-loop implementation.
) -> AdanOptimiOptimizer:
    "A partial function for the Adan optimizer with low precision and foreach implementations"
    return partialler(Adan, beta1=beta1, beta2=beta2, beta3=beta3, eps=eps, wd=wd,
                      decouple_lr=decouple_lr, adam_wd=adam_wd, kahan_sum=kahan_sum,
                      foreach=foreach)


# %% ../../nbs/optimizer.ipynb 23
def RAdam(
    params: Listified[Tensor],  # Model parameters or parameter groups
    lr: float,  # Default learning rate
    mom: float = 0.9,  # Gradient moving average (β1) coefficient
    sqr_mom: float = 0.99,  # Gradient squared moving average (β2) coefficient
    eps: float = 1e-5,  # Added for numerical stability
    wd: float = 0.,  # Optional weight decay (decoupled or L2)
    decouple_wd: bool = True,  # Apply decoupled weight decay (RAdamW) or L2 regularization (RAdam)
    decouple_lr: bool = False,  # Apply fully decoupled weight decay
    kahan_sum: bool | None = None,  # More accurate parameter updates when training in low precision (float16 or bfloat16). If unspecified, automatically applies for low precision parameters.
    foreach: bool | None = None,  # Use faster ForEach implementation. If unspecified, tries to use foreach over for-loop implementation.
) -> RAdamOptimiOptimizer:
    "A fastai-compatible RAdam optimizer with low precision and foreach implementations"

    if OPTIMI:
        return RAdamOptimiOptimizer(params, lr=lr, mom=mom, sqr_mom=sqr_mom, wd=wd, eps=eps,
                                    decouple_wd=decouple_wd, decouple_lr=decouple_lr,
                                    kahan_sum=kahan_sum, foreach=foreach)
    else:
        raise ImportError('optimi package not found. Run `pip install torch-optimi`.')


# %% ../../nbs/optimizer.ipynb 24
def radam(
    mom: float = 0.9,  # Gradient moving average (β1) coefficient
    sqr_mom: float = 0.99,  # Gradient squared moving average (β2) coefficient
    eps: float = 1e-5,  # Added for numerical stability
    wd: float = 0.,  # Optional weight decay (decoupled or L2)
    decouple_wd: bool = True,  # Apply decoupled weight decay (RAdamW) or L2 regularization (RAdam)
    decouple_lr: bool = False,  # Apply fully decoupled weight decay
    kahan_sum: bool | None = None,  # More accurate parameter updates when training in low precision (float16 or bfloat16). If unspecified, automatically applies for low precision parameters.
    foreach: bool | None = None,  # Use faster ForEach implementation. If unspecified, tries to use foreach over for-loop implementation.
) -> RAdamOptimiOptimizer:
    "Partial function for the RAdam optimizer with low precision and foreach implementations"
    return partialler(RAdam, mom=mom, sqr_mom=sqr_mom, eps=eps, wd=wd, decouple_wd=decouple_wd,
                      decouple_lr=decouple_lr, kahan_sum=kahan_sum, foreach=foreach)


# %% ../../nbs/optimizer.ipynb 26
def Ranger(
    params: Listified[Tensor],  # Model parameters or parameter groups
    lr: float,  # Default learning rate
    mom: float = 0.95,  # Gradient moving average (β1) coefficient
    sqr_mom: float = 0.99,  # Gradient squared moving average (β2) coefficient
    eps: float = 1e-6,  # Added for numerical stability
    wd: float = 0.01,  # Optional weight decay (decoupled or L2)
    k: int = 6,  # How often to conduct Lookahead step
    alpha: float = 0.5,  # Slow weight moving average coefficient
    decouple_wd: bool = True,  # Apply decoupled weight decay (RangerW) or L2 regularization (Ranger)
    decouple_lr: bool = False,  # Apply fully decoupled weight decay
    kahan_sum: bool | None = None,  # More accurate parameter updates when training in low precision (float16 or bfloat16). If unspecified, automatically applies for low precision parameters. Unsupported for `eightbit=True`.
    foreach: bool | None = None  # Use faster ForEach implementation. If unspecified, tries to use foreach over for-loop implementation.
) -> RangerOptimiOptimizer:
    "Convenience method for `Lookahead` with `RAdam` with low precision and foreach implementations"

    if OPTIMI:
        return RangerOptimiOptimizer(params, lr=lr, mom=mom, sqr_mom=sqr_mom, eps=eps, wd=wd,
                                     k=k, alpha=alpha, decouple_wd=decouple_wd, decouple_lr=decouple_lr,
                                     kahan_sum=kahan_sum, foreach=foreach)
    else:
        raise ImportError('optimi package not found. Run `pip install torch-optimi`.')


# %% ../../nbs/optimizer.ipynb 27
def ranger(
    mom: float = 0.95,  # Gradient moving average (β1) coefficient
    sqr_mom: float = 0.99,  # Gradient squared moving average (β2) coefficient
    eps: float = 1e-6,  # Added for numerical stability
    wd: float = 0.01,  # Optional weight decay (decoupled or L2)
    k: int = 6,  # How often to conduct Lookahead step
    alpha: float = 0.5,  # Slow weight moving average coefficient
    decouple_wd: bool = True,  # Apply decoupled weight decay (RangerW) or L2 regularization (Ranger)
    decouple_lr: bool = False,  # Apply fully decoupled weight decay
    kahan_sum: bool | None = None,  # More accurate parameter updates when training in low precision (float16 or bfloat16). If unspecified, automatically applies for low precision parameters. Unsupported for `eightbit=True`.
    foreach: bool | None = None  # Use faster ForEach implementation. If unspecified, tries to use foreach over for-loop implementation.
) -> RangerOptimiOptimizer:
    "Partial function for the Ranger optimizer using RAdam with low precision and foreach implementations"
    return partialler(Ranger, mom=mom, sqr_mom=sqr_mom, eps=eps, wd=wd, k=k,
                      alpha=alpha, decouple_wd=decouple_wd, decouple_lr=decouple_lr,
                      kahan_sum=kahan_sum, foreach=foreach)


# %% ../../nbs/optimizer.ipynb 29
def StableAdamW(
    params: Listified[Tensor],  # Model parameters or parameter groups
    lr: float,  # Default learning rate
    mom: float = 0.9,  # Gradient moving average (β1) coefficient
    sqr_mom: float = 0.99,  # Gradient squared moving average (β2) coefficient
    eps: float = 1e-5,  # Added for numerical stability
    wd: float = 0.01,  # Optional weight decay
    decouple_lr: bool = False,  # Apply fully decoupled weight decay
    kahan_sum: bool | None = None,  # More accurate parameter updates when training in low precision (float16 or bfloat16). If unspecified, automatically applies for low precision parameters. Unsupported for `eightbit=True`.
    foreach: bool | None = None,  # Use faster ForEach implementation. If unspecified, tries to use foreach over for-loop implementation.
) -> StableAdamWOptimiOptimizer:
    "A fastai-compatible StableAdamW optimizer with low precision and foreach implementations"

    if OPTIMI:
        return StableAdamWOptimiOptimizer(params, lr=lr, mom=mom, sqr_mom=sqr_mom, eps=eps, wd=wd,
                                          decouple_lr=decouple_lr, kahan_sum=kahan_sum, foreach=foreach)
    else:
        raise ImportError('optimi package not found. Run `pip install torch-optimi`.')


# %% ../../nbs/optimizer.ipynb 30
def stableadamw(
    mom: float = 0.9,  # Gradient moving average (β1) coefficient
    sqr_mom: float = 0.99,  # Gradient squared moving average (β2) coefficient
    eps: float = 1e-5,  # Added for numerical stability
    wd: float = 0.01,  # Optional weight decay (decoupled or L2)
    decouple_lr: bool = False,  # Apply fully decoupled weight decay
    kahan_sum: bool | None = None,  # More accurate parameter updates when training in low precision (float16 or bfloat16). If unspecified, automatically applies for low precision parameters. Unsupported for `eightbit=True`.
    foreach: bool | None = None,  # Use faster ForEach implementation. If unspecified, tries to use foreach over for-loop implementation.
) -> StableAdamWOptimiOptimizer:
    "Partial function for the StableAdamW optimizer with low precision and foreach implementations"
    return partialler(StableAdamW, mom=mom, sqr_mom=sqr_mom, eps=eps, wd=wd,
                      decouple_lr=decouple_lr, kahan_sum=kahan_sum, foreach=foreach)

# %% ../../nbs/optimizer.ipynb 34
def Larc(
    params:Listified[Tensor], # Model parameters or parameter groups
    lr:float, # Default learning rate
    mom:float=0.9, # Gradient moving average (β1) coefficient
    clip:bool=True, # LARC if clip=True, LARS if clip=False
    trust_coeff:float=0.02, # Trust coeffiecnet for calculating layerwise LR
    eps:float=1e-8, # Added for numerical stability
    wd:float=0., # Optional weight decay (decoupled or L2)
    decouple_wd:bool=True, # Apply decoupled weight decay or L2 regularization. Ignored if `eightbit=True`
    eightbit:bool=False, # Use fused 8-bit implementation. Only supports LARS: `clip=False`
    hide_warning:bool=False, # Hide warning
    **eightbitargs
) -> Optimizer|LARS8bitOptimizer:
    "A fastai LARC/LARS optimizer with eight-bit implementations"
    if eightbit:
        if EIGHTBIT:
            if clip:
                raise NotImplementedError(f'{eightbit=} only supports the LARS optimizer. Set `clip=False`.')
            if decouple_wd and wd > 0:
                raise NotImplementedError(f'8-bit LARS only supports L2 weight decay: {decouple_wd=}')
            return LARS8bitOptimizer(params, lr=lr, mom=mom, wd=wd, trust_coeff=trust_coeff, **eightbitargs)
        else:
            raise ImportError(f'{eightbit=}. bitsandbytes package not found. Run `pip install bitsandbytes`.')
    else:
        if not hide_warning:
            warn("fastxtend doesn't have a non-eight-bit Lamb implementation, using the"
                 " fastai implementation. Pass `hide_warning=True` to hide this message.")
        cbs = [weight_decay] if decouple_wd else [l2_reg]
        if mom!=0.: cbs.append(average_grad)
        cbs += [partial(larc_layer_lr, clip=clip), larc_step]
        return Optimizer(params, cbs, lr=lr, mom=mom, trust_coeff=trust_coeff, eps=eps, wd=wd)

# %% ../../nbs/optimizer.ipynb 35
def larc(
    mom:float=0.9, # Gradient moving average (β1) coefficient
    clip:bool=True, # LARC if clip=True, LARS if clip=False
    trust_coeff:float=0.02, # Trust coeffiecnet for calculating layerwise LR
    eps:float=1e-8, # Added for numerical stability
    wd:float=0., # Optional weight decay (decoupled or L2)
    decouple_wd:bool=True, # Apply decoupled weight decay or L2 regularization
    eightbit:bool=False, # Use fused 8-bit implementation. Only supports LARS
    hide_warning:bool=False, # Hide warning
    **eightbitargs
) -> Optimizer|LARS8bitOptimizer:
    "Partial function for the LARC/LARS optimizer with fused TorchScript & 8-bit implementations"
    return partialler(Larc, mom=mom, clip=clip, eps=eps, trust_coeff=trust_coeff,
                      wd=wd, decouple_wd=decouple_wd, eightbit=eightbit, hide_warning=hide_warning)

# %% ../../nbs/optimizer.ipynb 37
def Lamb(
    params:Listified[Tensor], # Model parameters or parameter groups
    lr:float, # Default learning rate
    mom:float=0.9, # Gradient moving average (β1) coefficient
    sqr_mom:float=0.99, # Gradient squared moving average (β2) coefficient
    eps:float=1e-5, # Added for numerical stability
    wd:float=0., # Optional weight decay (decoupled or L2)
    decouple_wd:bool=True, # Apply decoupled weight decay or L2 regularization. Ignored if `eightbit=True`
    eightbit:bool=False, # Use fused 8-bit implementation. Only supports Decoupled weight decay
    hide_warning:bool=False, # Hide warning
    **eightbitargs
) -> Optimizer|LAMB8bitOptimizer:
    "A fastai LAMB optimizer with fused ForEach, TorchScript, & 8-bit implementations"
    if eightbit:
        if EIGHTBIT:
            if parse(bitsandbytes.__version__) <= parse('0.43.1') and not hide_warning:
                raise ValueError("8-bit LAMB in bitsandbytes will error out weights too small to quantize. "
                                 "Pass `hide_warning=True` to ignore and use anyway.")
            if not decouple_wd and wd > 0:
                raise NotImplementedError(f'8-bit LAMB only supports Decoupled weight decay: {decouple_wd=}')
            return LAMB8bitOptimizer(params, lr=lr, mom=mom, sqr_mom=sqr_mom,
                                     eps=eps, wd=wd, **eightbitargs)
        else:
            raise ImportError(f'{eightbit=}. bitsandbytes package not found. Run `pip install bitsandbytes`.')
    else:
        if not hide_warning:
            warn("fastxtend doesn't have a non-eight-bit Lamb implementation, using the"
                 " fastai implementation. Pass `hide_warning=True` to hide this message.")
        cbs = [weight_decay] if decouple_wd else [l2_reg]
        cbs += [partial(average_grad, dampening=True), average_sqr_grad, step_stat, lamb_step]
        return Optimizer(params, cbs, lr=lr, mom=mom, sqr_mom=sqr_mom, eps=eps, wd=wd)

# %% ../../nbs/optimizer.ipynb 38
def lamb(
    mom:float=0.9, # Gradient moving average (β1) coefficient
    sqr_mom:float=0.99, # Gradient squared moving average (β2) coefficient
    eps:float=1e-5, # Added for numerical stability
    wd:float=0., # Optional weight decay (decoupled or L2)
    decouple_wd:bool=True, # Apply decoupled weight decay or L2 regularization
    eightbit:bool=False, # Use fused 8-bit implementation. Only supports Decoupled weight decay
    hide_warning:bool=False, # Hide warning
    **eightbitargs
) -> Optimizer|LAMB8bitOptimizer:
    "Partial function for the LAMB optimizer with fused ForEach, TorchScript, & 8-bit implementations"
    return partialler(Lamb, mom=mom, sqr_mom=sqr_mom, eps=eps, wd=wd, decouple_wd=decouple_wd,
                      eightbit=eightbit, hide_warning=hide_warning, **eightbitargs)

# %% ../../nbs/optimizer.ipynb 40
def RMSProp(
    params:Listified[Tensor], # Model parameters or parameter groups
    lr:float, # Default learning rate
    mom:float=0., # Gradient moving average (β1) coefficient
    sqr_mom:float=0.99, # Gradient squared moving average (β2) coefficient
    eps:float=1e-8, # Added for numerical stability
    wd:float=0., # Optional weight decay (true or L2)
    decouple_wd:bool=True, # Apply true weight decay or L2 regularization. Ignored if `eightbit=True`
    eightbit:bool=False, # Use fused 8-bit implementation. Only supports Decoupled weight decay
    hide_warning:bool=False, # Hide warning
    **eightbitargs
) -> Optimizer|RMSProp8bitOptimizer:
    "A fastai RMSProp/RMSPropW optimizer with fused TorchScript and 8-bit implementations"
    if eightbit:
        if EIGHTBIT:
            if decouple_wd and wd > 0:
                raise NotImplementedError(f'8-bit RMSProp only supports L2 weight decay: {decouple_wd=}')
            if mom > 0:
                raise NotImplementedError(f'8-bit RMSProp does not use momentum: {mom=}')
            return RMSProp8bitOptimizer(params, lr=lr, sqr_mom=sqr_mom, eps=eps, wd=wd, **eightbitargs)
        else:
            raise ImportError(f'{eightbit=}. bitsandbytes package not found. Run `pip install bitsandbytes`.')
    else:
        if not hide_warning:
            warn("fastxtend doesn't have a non-eight-bit RMSProp implementation, using the"
                 " fastai implementation. Pass `hide_warning=True` to hide this message.")
        cbs = [weight_decay] if decouple_wd else [l2_reg]
        cbs += ([average_sqr_grad] if mom==0. else [average_grad, average_sqr_grad])
        cbs.append(rms_prop_step)
        return Optimizer(params, cbs, lr=lr, mom=mom, sqr_mom=sqr_mom, wd=wd, eps=eps)

# %% ../../nbs/optimizer.ipynb 41
def rmsprop(
    mom:float=0., # Gradient moving average (β1) coefficient
    sqr_mom:float=0.99, # Gradient squared moving average (β2) coefficient
    eps:float=1e-8, # Added for numerical stability
    wd:float=0., # Optional weight decay (decoupled or L2)
    decouple_wd:bool=True, # Apply decoupled weight decay or L2 regularization
    eightbit:bool=False, # Use fused 8-bit implementation
    hide_warning:bool=False, # Hide warning
    **eightbitargs
) -> Optimizer|RMSProp8bitOptimizer:
    "Partial function for the RMSProp/RMSPropW optimizer with fused TorchScript and 8-bit implementations"
    return partialler(RMSProp, mom=mom, sqr_mom=sqr_mom, eps=eps, wd=wd,
                      decouple_wd=decouple_wd, eightbit=eightbit,
                      hide_warning=hide_warning, **eightbitargs)

# %% ../../nbs/optimizer.ipynb 44
def QHAdam(
    params:Listified[Tensor], # Model parameters or parameter groups
    lr:float, # Default learning rate
    mom:float=0.999, # Gradient moving average (β1) coefficient
    sqr_mom:float=0.999, # Gradient squared moving average (β2) coefficient
    nu_1:float=0.7, # QH immediate discount factor
    nu_2:float=1.0, # QH momentum discount factor
    eps:float=1e-8, # Added for numerical stability
    wd:float=0., # Optional weight decay (decoupled or L2)
    decouple_wd:bool=True, # Apply decoupled weight decay (QHAdamW) or L2 regularization (QHAdam)
    hide_warning:bool=False, # Hide warning
) -> Optimizer:
    "The fastai QHAdam/QHAdamW optimizer"
    if not hide_warning:
        warn("fastxtend doesn't have a QHAdam implementation, using the fastai"
             " implementation. Pass `hide_warning=True` to hide this message.")
    cbs = [weight_decay] if decouple_wd else [l2_reg]
    cbs += [partial(average_grad, dampening=True), average_sqr_grad, step_stat, qhadam_step]
    return Optimizer(params, cbs, lr=lr, nu_1=nu_1, nu_2=nu_2, mom=mom, sqr_mom=sqr_mom, eps=eps, wd=wd)

# %% ../../nbs/optimizer.ipynb 45
def qhadam(
    mom:float=0.999, # Gradient moving average (β1) coefficient
    sqr_mom:float=0.999, # Gradient squared moving average (β2) coefficient
    nu_1:float=0.7, # QH immediate discount factor
    nu_2:float=1.0, # QH momentum discount factor
    eps:float=1e-8, # Added for numerical stability
    wd:float=0., # Optional weight decay (decoupled or L2)
    decouple_wd:bool=True, # Apply decoupled weight decay (QHAdamW) or L2 regularization (QHAdam)
    hide_warning:bool=False, # Hide warning
) -> Optimizer:
    "Partial function for the fastai QHAdam/QHAdamW optimizer"
    return partialler(QHAdam, mom=mom, sqr_mom=sqr_mom, nu_1=nu_1, nu_2=nu_2, eps=eps,
                      wd=wd, decouple_wd=decouple_wd, hide_warning=hide_warning)