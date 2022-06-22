# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/callback.simpleprofiler.ipynb (unless otherwise specified).

__all__ = ['SimpleProfilerPostCallback', 'SimpleProfilerCallback']

# Cell
import locale
import time
import pandas as pd
import numpy as np
from pathlib import Path
from packaging.version import parse

from fastcore.foundation import docs
from fastcore.basics import mk_class, noop, in_notebook

import fastai
from fastai.learner import Learner, Recorder
from fastai.callback.core import *

from ..imports import *

if in_notebook():
    from IPython.display import display

# Cell
if parse(fastai.__version__) >= parse('2.7.0'):
    _inner_loop = "before_draw before_batch after_pred after_loss before_backward after_cancel_backward after_backward before_step after_step after_cancel_batch after_batch".split()
else:
    _inner_loop = "before_draw before_batch after_pred after_loss before_backward before_step after_step after_cancel_batch after_batch".split()

# Cell
if parse(fastai.__version__) >= parse('2.7.0'):
    _events = L.split('after_create before_fit before_epoch before_train before_draw before_batch after_pred after_loss \
        before_backward after_cancel_backward after_backward before_step after_cancel_step after_step \
        after_cancel_batch after_batch after_cancel_train after_train before_validate after_cancel_validate \
        after_validate after_cancel_epoch after_epoch after_cancel_fit after_fit')
else:
    _events = L.split('after_create before_fit before_epoch before_train before_draw before_batch after_pred after_loss \
        before_backward before_step after_cancel_step after_step after_cancel_batch after_batch after_cancel_train \
        after_train before_validate after_cancel_validate after_validate after_cancel_epoch \
        after_epoch after_cancel_fit after_fit')

mk_class('event', **_events.map_dict(),
         doc="All possible events as attributes to get tab-completion and typo-proofing")

# Cell
@patch
def __call__(self:Callback, event_name):
    "Call `self.{event_name}` if it's defined"
    _run = (event_name not in _inner_loop or (self.run_train and getattr(self, 'training', True)) or
            (self.run_valid and not getattr(self, 'training', False)))
    res = None
    if self.run and _run:
        try: res = getattr(self, event_name, noop)()
        except (CancelBatchException, CancelEpochException, CancelFitException, CancelStepException, CancelTrainException, CancelValidException): raise
        except Exception as e:
            e.args = [f'Exception occured in `{self.__class__.__name__}` when calling event `{event_name}`:\n\t{e.args[0]}']
            raise
    if event_name=='after_fit': self.run=True #Reset self.run to True at each end of fit
    return res

# Cell
@patch
def _call_one(self:Learner, event_name):
    if not hasattr(event, event_name): raise Exception(f'missing {event_name}')
    for cb in self.cbs.sorted('order'): cb(event_name)

# Cell
@patch
def all_batches(self:Learner):
    self.n_iter = len(self.dl)
    if hasattr(self, 'simple_profiler'):
        self.it = iter(self.dl)
        for i in range(self.n_iter):
            self("before_draw")
            self.one_batch(i, next(self.it))
        del(self.it)
    else:
        for o in enumerate(self.dl): self.one_batch(*o)

# Cell
_loop = ['Start Fit', 'before_fit', 'Start Epoch Loop', 'before_epoch', 'Start Train', 'before_train',
         'Start Batch Loop', 'before_draw', 'before_batch', 'after_pred', 'after_loss', 'before_backward',
         'before_step', 'after_step', 'after_cancel_batch', 'after_batch','End Batch Loop', 'End Train',
         'after_cancel_train', 'after_train', 'Start Valid', 'before_validate', 'Start Batch Loop',
         '**CBs same as train batch**', 'End Batch Loop', 'End Valid', 'after_cancel_validate',
         'after_validate', 'End Epoch Loop', 'after_cancel_epoch', 'after_epoch', 'End Fit',
         'after_cancel_fit', 'after_fit']

# Internal Cell
@patch
def show_training_loop(self:Learner):
    indent = 0
    for s in _loop:
        if s.startswith('Start'): print(f'{" "*indent}{s}'); indent += 2
        elif s.startswith('End'): indent -= 2; print(f'{" "*indent}{s}')
        else: print(f'{" "*indent} - {s:15}:', self.ordered_cbs(s))

# Internal Cell
_phase = ['fit', 'epoch', 'train', 'valid']
_epoch = ['train', 'valid']
_train = ['draw', 'batch', 'forward', 'loss', 'backward', 'opt_step', 'zero_grad']
_valid = ['draw', 'batch', 'predict', 'loss']

# Cell
class SimpleProfilerPostCallback(Callback):
    "Pair with `SimpleProfilerCallback` to profile training performance. Removes itself after training is over."
    order,remove_on_fetch = Recorder.order-1,True
    def __init__(self, samples_per_second=True):
        store_attr()
        self._phase,self._train,self._valid = _phase,_train,_valid

    def before_fit(self):
        self.profiler = self.learn.simple_profiler
        self.has_logger = self.profiler.has_logger
        self.n_train_batches = len(self.dls.train)
        self.n_valid_batches = len(self.dls.valid)

    def after_train(self):
        self.profiler._raw_values['train'].append(time.perf_counter() - self.profiler._train_start)

    def after_validate(self):
        self.profiler._raw_values['valid'].append(time.perf_counter() - self.profiler._validate_start)

    def after_pred(self):
        if self.training: self.profiler._raw_values['train_forward'].append(time.perf_counter() - self.profiler._train_batch_start)
        else:             self.profiler._raw_values['valid_predict'].append(time.perf_counter() - self.profiler._valid_batch_start)

        if self.training: self.profiler._train_loss_start = time.perf_counter()
        else:             self.profiler._valid_loss_start = time.perf_counter()

    def after_loss(self):
        if self.training: self.profiler._raw_values['train_loss'].append(time.perf_counter() - self.profiler._train_loss_start)
        else:             self.profiler._raw_values['valid_loss'].append(time.perf_counter() - self.profiler._valid_loss_start)

    def after_step(self):
        self.profiler._raw_values['train_opt_step'].append(time.perf_counter() - self.profiler._step_start)
        self.profiler._zero_start = time.perf_counter()

    def after_batch(self):
        if self.training:
            self.profiler._raw_values['train_batch'].append(time.perf_counter() - self.profiler._train_batch_start)
            if self.samples_per_second:
                self.profiler._raw_values['train_bs'].append(find_bs(self.learn.xb))
                if self.has_logger:
                    self.profiler._log_after_batch()
        else:
            self.profiler._raw_values['valid_batch'].append(time.perf_counter() - self.profiler._valid_batch_start)
            if self.samples_per_second:
                self.profiler._raw_values['valid_bs'].append(find_bs(self.learn.xb))

    def after_epoch(self):
        self.profiler._raw_values['epoch'].append(time.perf_counter() - self.profiler._epoch_start)

    def after_fit(self):
        self.profiler._raw_values['fit'].append(time.perf_counter() - self.profiler._fit_start)
        self.profiler._generate_report()
        if self.has_logger: self.profiler._log_after_fit()
        if not hasattr(self.learn, 'lr_finder'):
            self.profiler._display_report()
            self.learn.remove_cbs([SimpleProfilerCallback, SimpleProfilerPostCallback])

# Cell
class SimpleProfilerCallback(Callback):
    """
    Adds a simple profiler to the fastai `Learner`. Optionally showing formatted report or saving unformatted results as csv.

    Pair with SimpleProfilerPostCallback to profile training performance.

    Post fit, access report & results via `Learner.simple_profile_report` & `Learner.simple_profile_results`.
    """
    order,remove_on_fetch = TrainEvalCallback.order+1,True
    def __init__(self,
        show_report=True, # Display formatted report post profile
        plain=False, # For Jupyter Notebooks, display plain report
        markdown=False, # Display markdown formatted report
        save_csv=False,  # Save raw results to csv
        csv_name='simple_profile.csv', # CSV save location
        logger_callback='wandb' # Log report and samples/second to `logger_callback` using `Callback.name`
    ):
        store_attr()
        self.csv_name = Path(csv_name)
        self._phase,self._train,self._valid = _phase,_train,_valid
        self._log_after_batch = getattr(self, f'_{self.logger_callback}_log_after_batch', noop)
        self._log_after_fit   = getattr(self, f'_{self.logger_callback}_log_after_fit', noop)

    def before_fit(self):
        self.has_logger = hasattr(self.learn, self.logger_callback) and not hasattr(self.learn, 'lr_finder') and not hasattr(self, "gather_preds")
        self._raw_values = dict()
        for p in _phase:
            self._raw_values[p] = []
        for p in _epoch:
            for a in getattr(self, f'_{p}'):
                self._raw_values[f'{p}_{a}'] = []
            self._raw_values[f'{p}_bs'] = []

        self._fit_start = time.perf_counter()

    def before_epoch(self):
        self._epoch_start = time.perf_counter()

    def before_train(self):
        self._train_start = time.perf_counter()

    def before_validate(self):
        self._validate_start = time.perf_counter()

    def before_draw(self):
        if self.training: self._train_draw_start = time.perf_counter()
        else: self._valid_draw_start = time.perf_counter()

    def before_batch(self):
        if self.training: self._raw_values['train_draw'].append(time.perf_counter() - self._train_draw_start)
        else:             self._raw_values['valid_draw'].append(time.perf_counter() - self._valid_draw_start)

        if self.training: self._train_batch_start = time.perf_counter()
        else:             self._valid_batch_start = time.perf_counter()

    def before_backward(self):
        self._backward_start = time.perf_counter()

    def before_step(self):
        self._raw_values['train_backward'].append(time.perf_counter() - self._backward_start)
        self._step_start = time.perf_counter()

    def after_batch(self):
        if self.training: self._raw_values['train_zero_grad'].append(time.perf_counter() - self._zero_start)

    def _train_samples_per_second(self, action):
        if action =='draw':
            bs = self._raw_values['train_bs'][-1]
            batch = self._raw_values['train_batch'][-1]
            return -(bs/batch - bs/(batch+self._raw_values[f'train_draw'][-1]))
        else:
            return self._raw_values[f'train_bs'][-1]/self._raw_values[f'train_{action}'][-1]

    def _generate_report(self):
        total_time = self._raw_values['fit'][0]
        self.report = pd.DataFrame(columns=['Phase', 'Action', 'Step', 'Mean Duration', 'Duration Std Dev',
                                            'Number of Calls', 'Samples/Second', 'Total Time', 'Percent of Total'])

        for p in _phase:
            if p == 'fit':
                self._append_to_df(['fit', p, p, 0, 0, 1, '-', total_time, f'{self._calc_percent(total_time):.0%}'])
            else:
                if p in _epoch:
                    self._append_to_df(self._create_overview_row('fit', 'epoch', p, self._raw_values[p], np.array(self._raw_values[f'{p}_bs'])))
                else:
                    self._append_to_df(self._create_overview_row('fit', p, p, self._raw_values[p], None))

        for p in _epoch:
            bs = np.array(self._raw_values[f'{p}_bs'])
            for i, s in enumerate(getattr(self, f'_{p}')):
                if s in ['draw', 'batch']: a = s
                self._append_to_df(self._create_detail_row(p, a, s, self._raw_values[f'{p}_{s}'], bs if s in _train else None))

        self.learn.simple_profile_results = self.report.copy()
        for c in ['Mean Duration', 'Duration Std Dev', 'Total Time']:
            self.report[c] = self.report[c].apply(self._scale)
        self.report[['Phase', 'Action']] = self.report[['Phase', 'Action']].where(~self.report[['Phase', 'Action']].duplicated(), '')
        self.report['Phase']  = self.report['Phase'].where(~self.report['Phase'].duplicated(), '')
        self.report['Step']   = self.report['Step'].where(self.report['Step'] != self.report['Action']).fillna('')
        self.report['Action'] = self.report['Action'].where(self.report['Phase'] != self.report['Action']).fillna('')

        self.learn.simple_profile_report = self.report

    def _display_report(self):
        if self.show_report:
            if self.markdown: print(self.report.to_markdown(index=False))
            else:
                if in_notebook() and not self.plain:
                    with pd.option_context('display.max_rows', len(self.report.index)):
                        s = self.report.style.set_caption("Simple Profiler Results").hide_index()
                        display(s)
                else:
                    print('Simple Profiler Results')
                    print(self.report.to_string(index=False))

        if self.save_csv:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.learn.simple_profile_results.to_csv(self.path/self.csv_name, index=False)

    def _append_to_df(self, row):
        self.report.loc[len(self.report.index)] = row

    def _calc_percent(self, time):
        return time / self._raw_values['fit'][0]

    def _create_overview_row(self, phase, action, step, input, bs=None):
        if bs is not None:
            draw = np.array(self._raw_values[f'{step}_draw'])
            batch = np.array(self._raw_values[f'{step}_batch'])
            sam_per_sec = f'{int(np.around(np.mean(bs/(draw+batch)))):,d}'
        else:
            sam_per_sec = '-'
        return [phase, action, step, np.mean(input), np.std(input), len(input), sam_per_sec,
                np.sum(input), f'{self._calc_percent(np.sum(input)):.0%}']

    def _create_detail_row(self, phase, action, step, input, bs=None):
        if bs is None or step=='zero_grad': sam_per_sec = '-'
        elif action=='draw':
            batch = np.array(self._raw_values[f'{phase}_batch'])
            sam_per_sec = f'{-int(np.around(np.mean(bs/batch - bs/(np.array(input)+batch)))):,d}'
        else:
            sam_per_sec = f'{int(np.around(np.mean(bs/np.array(input)))):,d}'
        return [phase, action, step, np.mean(input), np.std(input), len(input), sam_per_sec,
                np.sum(input), f'{self._calc_percent(np.sum(input)):.0%}']

    # modified from https://github.com/thomasbrandon/mish-cuda/blob/master/test/perftest.py
    def _scale(self, val, spec="#0.4G"):
        if val == 0: return '-'
        PREFIXES = np.array([c for c in u"yzafpnµm kMGTPEZY"])
        exp = np.int8(np.log10(np.abs(val)) // 3 * 3 * np.sign(val))
        val /= 10.**exp
        prefix = PREFIXES[exp//3 + len(PREFIXES)//2]
        return f"{val:{spec}}{prefix}s"

# Cell
@patch
def profile(self:Learner,
        show_report=True, # Display formatted report post profile
        plain=False, # For Jupyter Notebooks, display plain report
        markdown=False, # Display markdown formatted report
        save_csv=False,  # Save raw results to csv
        csv_name='simple_profile.csv', # CSV save location
        samples_per_second=True, # Log samples/second for all actions & steps
        logger_callback='wandb' # Log report and samples/second to `logger_callback` using `Callback.name`
    ):
    "Run Simple Profiler when training. Simple Profiler removes itself when finished."
    self.add_cbs([SimpleProfilerCallback(show_report, plain, markdown, save_csv, csv_name, logger_callback),
                  SimpleProfilerPostCallback(samples_per_second)])
    return self

# Internal Cell
try:
    import wandb

    @patch
    def _wandb_log_after_batch(self:SimpleProfilerCallback):
        train_vals = {f'samples_per_second/train_{action}': self._train_samples_per_second(action) for action in _train[:-1]}
        wandb.log(train_vals, self.learn.wandb._wandb_step+1)

    @patch
    def _wandb_log_after_fit(self:SimpleProfilerCallback):
        report = wandb.Table(dataframe=self.learn.simple_profile_report)
        results = wandb.Table(dataframe=self.learn.simple_profile_results)

        wandb.log({"simple_profile_report": report})
        wandb.log({"simple_profile_results": results})
        wandb.log({}) # ensure sync
except:
    pass