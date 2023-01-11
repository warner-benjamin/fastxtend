# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/schedulers.ipynb.

# %% ../nbs/schedulers.ipynb 1
# Contains code from:
# fastai - Apache License 2.0 - Copyright (c) 2023 fast.ai

# %% ../nbs/schedulers.ipynb 3
from __future__ import annotations

import numpy as np

from fastcore.basics import even_mults

from fastai.callback.core import Callback
from fastai.callback.schedule import SchedCos, SchedNo, combine_scheds, ParamScheduler, _Annealer
from fastai.learner import Learner

from .basics import is_listish
from .imports import *

# %% auto 0
__all__ = []

# %% ../nbs/schedulers.ipynb 6
@patch
def fit_flat_warmup(self:Learner,
    n_epoch:int, # Number of epochs
    lr:float|None=None, # Maximum learning rate
    div:Numeric=25., # Initial learning rate: `lr/div`
    div_final:Numeric=1e5, # Final learning rate: `lr/div_final`
    pct_start:float=0.75, # Start learning rate cosine annealing
    warm_pct:float=0.2, # Learning rate warmup in percent
    warm_epoch:int=5, # Learning rate warmup in epochs
    warm_mode:str='auto', # Warmup using 'epoch', 'pct', or min of epoch/pct if 'auto'
    warm_sched:Callable[..., _Annealer]=SchedCos, # Learning rate warmup schedule
    wd:float|None=None, # Weight decay, defaults to `Optimizer` weight decay
    cbs:Listified[Callback]|None=None, # Temporary Callbacks to apply during fit
    reset_opt:bool=False # Reset `Optimizer` before fit
):
    "Fit `self.model` for `n_epoch` at flat `lr` with a warmup and ending with cosine annealing."
    if self.opt is None:
        self.create_opt()
    self.opt.set_hyper('lr', self.lr if lr is None else lr)

    if warm_mode=='auto':
        warm_pct = min(warm_epoch/n_epoch, warm_pct)
    elif warm_mode=='epoch':
        warm_pct = warm_epoch/n_epoch

    pcts = [warm_pct, 1-(warm_pct+(1-pct_start)), 1-pct_start]
    scheds = [warm_sched(lr/div, lr), SchedNo(lr,lr), SchedCos(lr, lr/div_final)]
    scheds = {'lr': combine_scheds(pcts, scheds)}

    self.fit(n_epoch, cbs=ParamScheduler(scheds)+L(cbs), reset_opt=reset_opt, wd=wd, start_epoch=0)

# %% ../nbs/schedulers.ipynb 11
@patch
def fit_cos_anneal(self:Learner,
    n_epoch:int, # Number of epochs
    lr:float|None=None, # Maximum learning rate
    div:Numeric=25., # Initial learning rate: `lr/div`
    div_final:Numeric=1e5, # Final learning rate: `lr/div_final`
    warm_pct:float=0.2, # Learning rate warmup in percent
    warm_epoch:int=5, # Learning rate warmup in epochs
    warm_mode:str='auto', # Warmup using 'epoch', 'pct', or min of epoch/pct 'auto'
    warm_sched:Callable[..., _Annealer]=SchedCos, # Learning rate warmup schedule
    wd:float|None=None, # Weight decay, defaults to `Optimizer` weight decay
    cbs:Listified[Callback]|None=None, # Temporary Callbacks to apply during fit
    reset_opt:bool=False # Reset `Optimizer` before fit
):
    "Fit `self.model` for `n_epoch` using a with cosine annealing schedule with a max `lr` and optional warmup."
    if self.opt is None:
        self.create_opt()
    self.opt.set_hyper('lr', self.lr if lr is None else lr)

    if warm_mode=='auto':
        warm_pct = min(warm_epoch/n_epoch, warm_pct)
    elif warm_mode=='epoch':
        warm_pct = warm_epoch/n_epoch

    if warm_pct > 0:
        pcts = [warm_pct, 1-warm_pct]
        scheds = [warm_sched(lr/div, lr), SchedCos(lr, lr/div_final)]
    else:
        pcts = [1]
        scheds = [SchedCos(lr, lr/div_final)]
    scheds = {'lr': combine_scheds(pcts, scheds)}

    self.fit(n_epoch, cbs=ParamScheduler(scheds)+L(cbs), reset_opt=reset_opt, wd=wd, start_epoch=0)

# %% ../nbs/schedulers.ipynb 19
@patch
def fit_flat_varied(self:Learner,
    n_epoch:int, # Number of epochs
    start_lr:float|None=None, # Initial learning rate
    div_final:Numeric=1e5, # Final learning rate: `lr/div_final`
    pct_start:float=0.75, # Start learning rate cosine annealing
    wd:float|None=None, # Weight decay, defaults to `Optimizer` weight decay
    next_lr:Listified[float]|slice|None=None, # Learning rates to switch to at `change_by`. Must be same length as `change_by`
    change_by:Listified[int]|Listified[float]|None=None, # Epochs or percent of steps to switch to `next_lr` by. Must be same length as `next_lr`
    change_time:Listified[int]|Listified[float]=1, # If greater than 0 (percent of steps or epochs), how long to change to `next_lr`. Must be same length as `next_lr`
    change_sched:Listified[Callable[..., _Annealer]]|None=None, # Schedule(s) for change. Defaults to `SchedCos`. Must be same length as `next_lr`
    cbs:Listified[Callback]|None=None, # Temporary Callbacks to apply during fit
    reset_opt:bool=False # Reset `Optimizer` before fit
):
    """
    Fit `self.model` for `n_epoch` at flat `start_lr`, then change to flat `next_lr` at `change_by`,
    optionally with cosine annealing or custom `change_sched` over `change_time`. Final cosine annealing at `pct_start`.
    """
    assert isinstance(next_lr, (float, slice)) or (is_listish(next_lr) and len(next_lr)>=1), f'{next_lr=} must be float, slice, or list of float or slice'
    assert isinstance(change_by, (int, float, slice)) or (is_listish(change_by) and len(change_by)>=1), f'{change_by=} must be int, float, slice, or list of int, float, or slice'

    if self.opt is None:
        self.create_opt()
    self.opt.set_hyper('lr', self.lr if start_lr is None else start_lr)
    start_lr = np.array([h['lr'] for h in self.opt.hypers])
    params_len = len(start_lr)

    if not is_listish(next_lr):
        next_lr = [next_lr]
    if not is_listish(change_by):
        change_by = [change_by]
    change_by = [i/n_epoch if i>=1 else i for i in change_by]
    assert len(change_by)==len(next_lr), f'{next_lr=} & {change_by=} need to be same length'

    if not is_listish(change_time):
        change_time = [change_time]*len(change_by)
    else: assert len(change_by)==len(change_time), f'{change_time=} list needs to be same length as {next_lr=} & {change_by=}'
    change_time = [i/n_epoch if i>=1 else i for i in change_time]

    if change_sched is not None:
        if not is_listish(change_sched):
            change_sched = [change_sched]
        assert len(change_by)==len(change_sched), f'{next_lr=} & {change_sched=} need to be same length'

    pcts, scheds, last_lr, last_pct = [], [SchedNo(start_lr, start_lr)], start_lr, 0
    for i, cb in enumerate(change_by):
        if cb < pct_start:
            nlr = next_lr[i]
            if isinstance(nlr, slice):
                if nlr.start: nlr = even_mults(nlr.start, nlr.stop, params_len)
                else: nlr = [nlr.stop/10]*(params_len-1) + [nlr.stop]
                nlr=np.array(nlr)

            change_pct = cb - change_time[i]
            assert change_pct >= last_pct, f'{change_pct=} in pos {i} of {change_by=} overlaps with previous schedule {last_pct}'

            pcts.append(change_pct - sum(pcts))
            scheds.append(SchedNo(nlr, nlr))

            if change_time[i] > 0:
                pcts.append(change_time[i])
                if is_listish(change_sched): scheds.insert(-1, change_sched[i](last_lr, nlr))
                else: scheds.insert(-1, SchedCos(last_lr, nlr))

            last_lr = nlr
            last_pct = change_pct
        else:
            warn(f'change_by: {change_by[i]} is after {pct_start=} and ignored.')

    pcts += [pct_start - sum(pcts), 1-pct_start]
    scheds += [SchedCos(last_lr, last_lr/div_final)]
    scheds = {'lr': combine_scheds(pcts, scheds)}

    self.fit(n_epoch, cbs=ParamScheduler(scheds)+L(cbs), reset_opt=reset_opt, wd=wd)
