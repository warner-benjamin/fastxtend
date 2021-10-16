import warnings
import numpy as np
from fastcore.basics import even_mults
from fastcore.foundation import patch, L
from fastai.learner import Learner
from fastai.callback.schedule import SchedCos, SchedNo, combine_scheds, ParamScheduler

def is_listish(x):
    return isinstance(x, (tuple,list,L))

@patch
def fit_flat_varied(self:Learner, n_epoch, start_lr=None, div_final=1e5, pct_start=0.75, wd=None,
                    next_lr=None, change_by=None, change_time=1, change_sched=None, cbs=None, reset_opt=False):
    """
    Fit `self.model` for `n_epoch` at flat `start_lr`, then change to flat `next_lr` at `change_by`, 
    optionally with cosine annealing or custom `change_sched` over `change_time`. Final cosine annealing at `pct_start`.
    
    n_epoch, start_lr, div_final, pct_start5, wd, cbs, & reset_opt are all same as fit_flat_cos from fast.ai.

    `next_lr` single or list of learning rates to switch to at change_by. Must be same length as `change_by`.
    `change_by` single or list of epochs or percent of steps to switch to `next_lr` by. Must be same length as `next_lr`.
    `change_time` if greater than 0 (percent of steps or epochs), how long to cosine anneal to `next_lr`. Can be single or list of same length as `next_lr`.
    `change_sched` optional single or list of fast.ai schedules. If `None` defaults to `SchedCos`. Must be same length as `next_lr`. `SchedPoly` must be passed as partial: `partial(SchedPoly, power=0.5)`.
    """
    assert isinstance(next_lr, (float, slice)) or (is_listish(next_lr) and len(next_lr)>=1), '`next_lr` must be float, slice, or list of float or slice'
    assert isinstance(change_by, (int, float, slice)) or (is_listish(change_by) and len(change_by)>=1), '`change_by` must be int, float, slice, or list of int, float, or slice'

    if self.opt is None: self.create_opt()
    self.opt.set_hyper('lr', self.lr if start_lr is None else start_lr)
    start_lr = np.array([h['lr'] for h in self.opt.hypers])
    params_len = len(start_lr)

    if not is_listish(next_lr): next_lr = [next_lr]
    if not is_listish(change_by): change_by = [change_by]
    change_by = [i/n_epoch if i>=1 else i for i in change_by]
    assert len(change_by)==len(next_lr), '`next_lr` & `change_by` need to be same length'

    if not is_listish(change_time): change_time = [change_time]*len(change_by)
    else: assert len(change_by)==len(change_time), '`change_time` list needs to be same length as `next_lr` & `change_by`'
    change_time = [i/n_epoch if i>=1 else i for i in change_time]

    if change_sched is not None: 
        if not is_listish(change_sched): change_sched = [change_sched]
        assert len(change_by)==len(change_sched), '`next_lr` & `change_sched` need to be same length'

    pcts, scheds, last_lr, last_pct = [], [SchedNo(start_lr, start_lr)], start_lr, 0
    for i, cb in enumerate(change_by):
        if cb < pct_start:
            nlr = next_lr[i]
            if isinstance(nlr, slice):
                if nlr.start: nlr = even_mults(nlr.start, nlr.stop, params_len)
                else: nlr = [nlr.stop/10]*(params_len-1) + [nlr.stop]
                nlr=np.array(nlr)
            change_pct = cb - change_time[i]
            assert change_pct >= last_pct, f'{change_pct} in pos {i} of `change_by` overlaps with previous schedule {last_pct}'
            pcts.append(change_pct - sum(pcts))
            scheds.append(SchedNo(nlr, nlr))
            if change_time[i] > 0:
                pcts.append(change_time[i])
                if is_listish(change_sched): scheds.insert(-1, change_sched[i](last_lr, nlr))
                else: scheds.insert(-1, SchedCos(last_lr, nlr))
            last_lr = nlr
            last_pct = change_pct
        else: warnings.warn(f'change_by: {change_by[i]} is after pct_start={pct_start} and ignored.')
    pcts += [pct_start - sum(pcts), 1-pct_start]
    scheds += [SchedCos(last_lr, last_lr/div_final)]

    scheds = {'lr': combine_scheds(pcts, scheds)}
    self.fit(n_epoch, cbs=ParamScheduler(scheds)+L(cbs), reset_opt=reset_opt, wd=wd)