import warnings
import numpy as np
from fastcore.utils import is_listy
from fastcore.foundation import patch, L
from fastai.learner import Learner
from fastai.callback.schedule import SchedCos, combine_scheds, ParamScheduler

@patch
def fit_flat_varied(self:Learner, n_epoch, start_lr=None, div_final=1e5, pct_start=0.75, wd=None,
                    next_lr=None, change_by=None, change_time=1, cbs=None, reset_opt=False):
    """
    Fit `self.model` for `n_epoch` at flat `start_lr`, then change to flat `next_lr` at `change_by`, 
    optionaly with cosine annealing over `change_time`. Final cosine annealing at `pct_start`.
    
    n_epoch, start_lr, div_final, pct_start5, wd, cbs, & reset_opt are all same as fit_flat_cos from fast.ai.

    `next_lr` single or list of learning rates to switch to at change_by. Must be same length as change_by.
    `change_by` single or list of epochs or percent of training to switch to next_lr by. Must be same length as next_lr.
    `change_time` if greater than 0 (pct of training or epochs), how long to cosine anneal to next_lr. Can single or list. 
    """
    assert isinstance(next_lr, (float, int)) or (is_listy(next_lr) and len(next_lr)>=1), '`next_lr` must be numeric or list of numeric'
    assert isinstance(change_by, (float, int)) or (is_listy(change_by) and len(change_by)>=1), '`change_by` must be numeric or list of numeric'

    if self.opt is None: self.create_opt()
    self.opt.set_hyper('lr', self.lr if start_lr is None else start_lr)
    lr = np.array([h['lr'] for h in self.opt.hypers])

    if not is_listy(next_lr): next_lr = [next_lr]
    if not is_listy(change_by): change_by = [change_by]
    change_by = [i/n_epoch if i>=1 else i for i in change_by]
    assert len(change_by)==len(next_lr), '`next_lr` & `change_by` lists need to be same length'

    if not is_listy(change_time): change_time = [change_time]*len(change_by)
    else: assert len(change_by)==len(change_time), '`change_time` list needs to be same length as `next_lr` & `change_by`'
    change_time = [i/n_epoch if i>=1 else i for i in change_time]

    pcts, cos_scheds, last_lr, last_pct = [], [SchedCos(start_lr, start_lr)], lr, 0
    for i, ra in enumerate(change_by):
        if ra < pct_start:
            rlr = next_lr[i]
            change_pct = ra - change_time[i]
            assert change_pct > last_pct, f'{change_pct} in pos {i} of `change_by` overlaps with previous schedule {last_pct}'
            pcts.append(change_pct - sum(pcts))
            cos_scheds.append(SchedCos(rlr, rlr))
            if change_time[i] > 0:
                pcts.append(change_time[i])
                cos_scheds.insert(-1, SchedCos(last_lr, rlr))       
            last_lr = rlr
            last_pct = change_pct
        else: warnings.warn(f'change_by: {change_by[i]} is after pct_start={pct_start} and ignored.')
    pcts += [pct_start - sum(pcts), 1-pct_start]
    cos_scheds += [SchedCos(last_lr, last_lr/div_final)]

    scheds = {'lr': combine_scheds(pcts, cos_scheds)}
    self.fit(n_epoch, cbs=ParamScheduler(scheds)+L(cbs), reset_opt=reset_opt, wd=wd)