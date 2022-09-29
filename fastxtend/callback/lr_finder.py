# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/callback.lr_finder.ipynb.

# %% auto 0
__all__ = ['LRFinder']

# %% ../../nbs/callback.lr_finder.ipynb 4
from fastcore.xtras import is_listy
from fastcore.foundation import patch, docs, Path
from fastcore.basics import tuplify
from fastai.callback.schedule import ParamScheduler, SchedExp, SuggestionMethod
from fastai.torch_core import tensor, get_random_states, set_random_states
from fastai.learner import Learner, CancelFitException, CancelValidException
from functools import partial
from copy import deepcopy
import torch
import collections, tempfile

# %% ../../nbs/callback.lr_finder.ipynb 5
@docs
class LRFinder(ParamScheduler):
    "Training with exponentially growing learning rate"
    def __init__(self, start_lr=1e-7, end_lr=10, num_it=100, stop_div=True, restore_state=True):
        if num_it < 6: num_it = 6
        self.scheds = {'lr': [SchedExp(s, e) for (s,e) in zip(start_lr,end_lr)
                             ] if is_listy(start_lr) else SchedExp(start_lr, end_lr)}
        self.num_it,self.stop_div,self.restore_state = num_it,stop_div,restore_state

    def before_fit(self):
        "Initialize container for hyper-parameters and save the model & optimizer, optionally saving dataloader & random state"
        super().before_fit()
        if self.restore_state:
            self.old_dls = deepcopy(self.learn.dls)
            self.states = get_random_states()
        path = self.path/self.model_dir
        path.mkdir(parents=True, exist_ok=True)
        self.tmp_d = tempfile.TemporaryDirectory(dir=path)
        self.tmp_p = Path(self.tmp_d.name).stem
        self.learn.save(f'{self.tmp_p}/_tmp')
        self.best_loss = float('inf')

    def before_batch(self):
        "Set the proper hyper-parameters in the optimizer"
        self._update_val(self.train_iter/self.num_it)

    def after_batch(self):
        "Record hyper-parameters of this batch and potentially stop training"
        super().after_batch()
        if self.smooth_loss < self.best_loss: self.best_loss = self.smooth_loss
        if self.smooth_loss > 4*self.best_loss and self.stop_div: raise CancelFitException()
        if self.train_iter >= self.num_it: raise CancelFitException()

    def before_validate(self):
        "Skip the validation part of training"
        raise CancelValidException()

    def after_fit(self):
        "Save the hyper-parameters in the recorder if there is one and load the original model & optimizer, optionally restoring dataloader & random state"
        self.learn.opt.zero_grad() # Needed before detaching the optimizer for future fits
        tmp_f = self.path/self.model_dir/self.tmp_p/'_tmp.pth'
        if tmp_f.exists():
            self.learn.load(f'{self.tmp_p}/_tmp', with_opt=True)
            self.tmp_d.cleanup()
        if self.restore_state:
            self.learn.dls = self.old_dls
            set_random_states(**self.states)

# %% ../../nbs/callback.lr_finder.ipynb 14
@patch
def lr_find(self:Learner, start_lr=1e-7, end_lr=10, num_it=100, stop_div=True, show_plot=True, suggest_funcs=(SuggestionMethod.Valley), restore_state=True):
    """
    Launch a mock training to find a good learning rate and return suggestions based on `suggest_funcs` as a named tuple. 
    
    Use `restore_state` to reset dataloaders and random state after running.
    """
    n_epoch = num_it//len(self.dls.train) + 1
    cb=LRFinder(start_lr=start_lr, end_lr=end_lr, num_it=num_it, stop_div=stop_div,restore_state=restore_state)
    with self.no_logging(): self.fit(n_epoch, cbs=cb)
    if suggest_funcs is not None:
        lrs, losses = tensor(self.recorder.lrs[num_it//10:-5]), tensor(self.recorder.losses[num_it//10:-5])
        nan_idxs = torch.nonzero(torch.isnan(losses.view(-1)))
        if len(nan_idxs) > 0:
            drop_idx = min(nan_idxs)
            lrs = lrs[:drop_idx]
            losses = losses[:drop_idx]
        _suggestions, nms = [], []
        for func in tuplify(suggest_funcs):
            nms.append(func.__name__ if not isinstance(func, partial) else func.func.__name__) # deal with partials
            _suggestions.append(func(lrs, losses, num_it))

        SuggestedLRs = collections.namedtuple('SuggestedLRs', nms)
        lrs, pnts = [], []
        for lr, pnt in _suggestions:
            lrs.append(lr)
            pnts.append(pnt)
        if show_plot: self.recorder.plot_lr_find(suggestions=pnts, nms=nms)
        return SuggestedLRs(*lrs)

    elif show_plot: self.recorder.plot_lr_find()
