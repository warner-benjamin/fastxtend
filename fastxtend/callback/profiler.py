# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/callback.profiler.ipynb.

# %% auto 0
__all__ = ['ThroughputCallback', 'ThroughputPostCallback', 'SimpleProfilerCallback', 'SimpleProfilerPostCallback', 'ProfileMode']

# %% ../../nbs/callback.profiler.ipynb 2
# Contains code from:
# fastai - Apache License 2.0 - Copyright (c) 2023 fast.ai

# %% ../../nbs/callback.profiler.ipynb 4
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
from ..utils import scale_time

if in_notebook():
    from IPython.display import display

# %% ../../nbs/callback.profiler.ipynb 10
if parse(fastai.__version__) >= parse('2.7.0'):
    _inner_loop = "before_draw before_batch after_pred after_loss before_backward after_cancel_backward after_backward before_step after_step after_cancel_batch after_batch".split()
else:
    _inner_loop = "before_draw before_batch after_pred after_loss before_backward before_step after_step after_cancel_batch after_batch".split()

# %% ../../nbs/callback.profiler.ipynb 11
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

# %% ../../nbs/callback.profiler.ipynb 13
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

# %% ../../nbs/callback.profiler.ipynb 15
@patch
def _call_one(self:Learner, event_name):
    if not hasattr(event, event_name): raise Exception(f'missing {event_name}')
    for cb in self.cbs.sorted('order'): cb(event_name)

# %% ../../nbs/callback.profiler.ipynb 17
@patch
def all_batches(self:Learner):
    self.n_iter = len(self.dl)
    self.it = iter(self.dl)
    for i in range(self.n_iter):
        self("before_draw")
        self.one_batch(i, next(self.it))
    del(self.it)

# %% ../../nbs/callback.profiler.ipynb 18
_loop = ['Start Fit', 'before_fit', 'Start Epoch Loop', 'before_epoch', 'Start Train', 'before_train',
         'Start Batch Loop', 'before_draw', 'before_batch', 'after_pred', 'after_loss', 'before_backward',
         'before_step', 'after_step', 'after_cancel_batch', 'after_batch','End Batch Loop', 'End Train',
         'after_cancel_train', 'after_train', 'Start Valid', 'before_validate', 'Start Batch Loop',
         '**CBs same as train batch**', 'End Batch Loop', 'End Valid', 'after_cancel_validate',
         'after_validate', 'End Epoch Loop', 'after_cancel_epoch', 'after_epoch', 'End Fit',
         'after_cancel_fit', 'after_fit']

# %% ../../nbs/callback.profiler.ipynb 19
@patch
def show_training_loop(self:Learner):
    indent = 0
    for s in _loop:
        if s.startswith('Start'): print(f'{" "*indent}{s}'); indent += 2
        elif s.startswith('End'): indent -= 2; print(f'{" "*indent}{s}')
        else: print(f'{" "*indent} - {s:15}:', self.ordered_cbs(s))

# %% ../../nbs/callback.profiler.ipynb 21
_phase = ['fit', 'epoch', 'train', 'valid']
_epoch = ['train', 'valid']
_train_full  = ['step', 'draw', 'batch', 'forward', 'loss', 'backward', 'opt_step', 'zero_grad']
_valid_full  = ['step', 'draw', 'batch', 'predict', 'loss']
_train_short = ['step', 'draw', 'batch']
_valid_short = _train_short

# %% ../../nbs/callback.profiler.ipynb 22
class ThroughputCallback(Callback):
    """
    Adds a throughput profiler to the fastai `Learner`. Optionally showing formatted report or saving unformatted results as csv.

    Pair with ThroughputPostCallback to profile training performance.

    Post fit, access report & results via `Learner.profile_report` & `Learner.profile_results`.
    """
    order,remove_on_fetch = TrainEvalCallback.order+1,True
    def __init__(self,
        show_report:bool=True, # Display formatted report post profile
        plain:bool=False, # For Jupyter Notebooks, display plain report
        markdown:bool=False, # Display markdown formatted report
        save_csv:bool=False,  # Save raw results to csv
        csv_name:str='throughput.csv', # CSV save location
        rolling_average:int=10, # Number of batches to average throughput over
        drop_first_batch:bool=True, # Drop the first batch from profiling
        logger_callback='wandb' # Log report and samples/second to `logger_callback` using `Callback.name`
    ):
        store_attr(but='csv_name,average,drop_first_batch')
        self.csv_name = Path(csv_name)
        self._drop = int(drop_first_batch)
        self._rolling_average = rolling_average
        self._log_after_batch = getattr(self, f'_{logger_callback}_log_after_batch', noop)
        self._log_after_fit   = getattr(self, f'_{logger_callback}_log_after_fit', noop)
        self._phase, self._train, self._valid = _phase, _train_short, _valid_short

    def before_fit(self):
        self.has_logger = hasattr(self.learn, self.logger_callback) and not hasattr(self.learn, 'lr_finder') and not hasattr(self, "gather_preds")
        self._raw_values, self._processed_samples = {}, {}
        for p in _phase:
            self._raw_values[p] = []
        for p in _epoch:
            for a in getattr(self, f'_{p}'):
                if a!='samples':
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
        if self.training:
            self._train_draw_start = time.perf_counter()
        else:
            self._valid_draw_start = time.perf_counter()

    def before_batch(self):
        if self.training:
            self._raw_values['train_draw'].append(time.perf_counter() - self._train_draw_start)
            self._train_batch_start = time.perf_counter()
        else:
            self._raw_values['valid_draw'].append(time.perf_counter() - self._valid_draw_start)
            self._valid_batch_start = time.perf_counter()

    def _samples_per_second(self, bs, action, epoch='train'):
        if action in ['step', 'draw']:
            batch = np.mean(self._raw_values[f'{epoch}_batch'][-self._rolling_average:])
            draw = np.mean(self._raw_values[f'{epoch}_draw'][-self._rolling_average:])
            return -((bs/batch if action=='draw' else 0) - bs/(draw+batch))
        else:
            return bs/np.mean(self._raw_values[f'{epoch}_{action}'][-self._rolling_average:])

    def _generate_report(self):
        total_time = self._raw_values['fit'][0]
        self.report = pd.DataFrame(columns=['Phase', 'Action', 'Mean Duration', 'Duration Std Dev',
                                            'Number of Calls', 'Samples/Second', 'Total Time', 'Percent of Total'])
        for p in _phase:
            if p == 'fit':
                self._append_to_df(['fit', p, 0, 0, 1, '-', total_time, f'{self._calc_percent(total_time):.0%}'])
            elif p == 'epoch':
                self._append_to_df(self._create_overview_row('fit', p, self._raw_values[p], None))
            else:
                self._append_to_df(self._create_overview_row('fit', p, self._raw_values[p], np.array(self._raw_values[f'{p}_bs'])))

        for p in _epoch:
            bs = np.array(self._raw_values[f'{p}_bs'])
            for a in getattr(self, f'_{p}'):
                if a == 'step':
                    self._raw_values[f'{p}_step'] = np.array(self._raw_values[f'{p}_draw']) + np.array(self._raw_values[f'{p}_batch'])
                    values = self._raw_values[f'{p}_step']
                else:
                    self._raw_values[f'{p}_{a}'] = np.array(self._raw_values[f'{p}_{a}'])
                    values = self._raw_values[f'{p}_{a}']
                self._append_to_df(self._create_detail_row(p, a, values, bs))

        self.learn.profile_results = self.report.copy()
        for c in ['Mean Duration', 'Duration Std Dev', 'Total Time']:
            self.report[c] = self.report[c].apply(scale_time)
        self.report[['Phase', 'Action']] = self.report[['Phase', 'Action']].where(~self.report[['Phase', 'Action']].duplicated(), '')
        self.report['Phase']  = self.report['Phase'].where(~self.report['Phase'].duplicated(), '')
        self.report['Action'] = self.report['Action'].where(self.report['Phase'] != self.report['Action']).fillna('')
        self.learn.profile_report = self.report

    def _display_report(self):
        if self.show_report:
            if self.markdown:
                print(self.report.to_markdown(index=False))
            else:
                if in_notebook() and not self.plain:
                    with pd.option_context('display.max_rows', len(self.report.index)):
                        s = self.report.style.set_caption("Profiling Results").hide(axis='index')
                        display(s)
                else:
                    print('\nProfiling Results:')
                    print(self.report.to_string(index=False))
            if self._drop > 0:
                print(f'Batch dropped. train and valid phases show {self._drop} less batch than fit.')
        if self.save_csv:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.learn.profile_results.to_csv(self.path/self.csv_name, index=False)

    def _append_to_df(self, row):
        self.report.loc[len(self.report.index)] = row

    def _calc_percent(self, time):
        return time / self._raw_values['fit'][0]

    def _create_overview_row(self, phase, action, input, bs=None):
        if bs is not None:
            draw = np.array(self._raw_values[f'{action}_draw'])
            batch = np.array(self._raw_values[f'{action}_batch'])
            self._processed_samples[f'{phase}_{action}'] = bs/(draw+batch)
            sam_per_sec = f'{int(np.around(self._processed_samples[f"{phase}_{action}"].mean())):,d}'
        else:
            sam_per_sec = '-'
        return [phase, action, np.mean(input), np.std(input), len(input), sam_per_sec,
                np.sum(input), f'{self._calc_percent(np.sum(input)):.0%}']

    def _create_detail_row(self, phase, action, input, bs=None):
        input = input[self._drop:]
        if bs is None or action=='zero_grad':
            sam_per_sec = '-'
        elif action == 'draw':
            bs = np.array(bs[self._drop:])
            batch = self._raw_values[f'{phase}_batch'][self._drop:]
            self._processed_samples[f'{phase}_{action}'] = -(bs/batch - bs/(input+batch))
            sam_per_sec = f'{int(np.around(self._processed_samples[f"{phase}_{action}"].mean())):,d}'
        else:
            bs = np.array(bs[self._drop:])
            self._processed_samples[f'{phase}_{action}'] = bs/input
            sam_per_sec = f'{int(np.around(self._processed_samples[f"{phase}_{action}"].mean())):,d}'
        return [phase, action, np.mean(input), np.std(input), len(input), sam_per_sec,
                np.sum(input), f'{self._calc_percent(np.sum(input)):.0%}']

# %% ../../nbs/callback.profiler.ipynb 23
class ThroughputPostCallback(Callback):
    "Required pair with `ThroughputCallback` to profile training performance. Removes itself after training is over."
    order,remove_on_fetch = Recorder.order-1,True
    def __init__(self):
        self._log_full = False
        self._phase, self._train, self._valid = _phase, _train_short, _train_short

    def before_fit(self):
        self.profiler = self.learn.throughput
        self.has_logger = self.profiler.has_logger
        self._start_train_logging, self._start_valid_logging = False, False
        self.n_train_batches = len(self.dls.train)
        self.n_valid_batches = len(self.dls.valid)
        self._rolling_average = self.profiler._rolling_average
        self._iter = -self.profiler._drop

    def after_train(self):
        self.profiler._raw_values['train'].append(time.perf_counter() - self.profiler._train_start)

    def after_validate(self):
        self.profiler._raw_values['valid'].append(time.perf_counter() - self.profiler._validate_start)

    def after_batch(self):
        if self.training:
            self.profiler._raw_values['train_batch'].append(time.perf_counter() - self.profiler._train_batch_start)
            self.profiler._raw_values['train_bs'].append(find_bs(self.learn.yb))
            if self.has_logger and self._iter >= self._rolling_average and self._iter % self._rolling_average == 0:
                self.profiler._log_after_batch(self._train)
            self._iter += 1
        else:
            self.profiler._raw_values['valid_batch'].append(time.perf_counter() - self.profiler._valid_batch_start)
            self.profiler._raw_values['valid_bs'].append(find_bs(self.learn.yb))

    def after_epoch(self):
        self.profiler._raw_values['epoch'].append(time.perf_counter() - self.profiler._epoch_start)

    def _after_fit(self, callbacks):
        self.profiler._raw_values['fit'].append(time.perf_counter() - self.profiler._fit_start)
        self.profiler._generate_report()
        if self.has_logger: self.profiler._log_after_fit()
        if not hasattr(self.learn, 'lr_finder'):
            self.profiler._display_report()
            self.learn.remove_cbs(callbacks)

    def after_fit(self):
        self._after_fit([ThroughputCallback, ThroughputPostCallback])

# %% ../../nbs/callback.profiler.ipynb 25
class SimpleProfilerCallback(ThroughputCallback):
    """
    Adds a simple profiler to the fastai `Learner`. Optionally showing formatted report or saving unformatted results as csv.

    Pair with SimpleProfilerPostCallback to profile training performance.

    Post fit, access report & results via `Learner.profile_report` & `Learner.profile_results`.
    """
    order,remove_on_fetch = TrainEvalCallback.order+1,True
    def __init__(self,
        show_report:bool=True, # Display formatted report post profile
        plain:bool=False, # For Jupyter Notebooks, display plain report
        markdown:bool=False, # Display markdown formatted report
        save_csv:bool=False,  # Save raw results to csv
        csv_name:str='simpleprofiler.csv', # CSV save location
        rolling_average:int=10, # Number of batches to average throughput over
        drop_first_batch:bool=True, # Drop the first batch from profiling
        logger_callback='wandb' # Log report and samples/second to `logger_callback` using `Callback.name`
    ):
        super().__init__(show_report=show_report, plain=plain, markdown=markdown, save_csv=save_csv,
                         csv_name=csv_name, rolling_average=rolling_average, drop_first_batch=drop_first_batch,
                         logger_callback=logger_callback)
        self._phase, self._train, self._valid = _phase, _train_full, _valid_full

    def before_backward(self):
        self._backward_start = time.perf_counter()

    def before_step(self):
        self._raw_values['train_backward'].append(time.perf_counter() - self._backward_start)
        self._step_start = time.perf_counter()

    def after_batch(self):
        if self.training:
            self._raw_values['train_zero_grad'].append(time.perf_counter() - self._zero_start)

# %% ../../nbs/callback.profiler.ipynb 26
class SimpleProfilerPostCallback(ThroughputPostCallback):
    "Required pair with `SimpleProfilerCallback` to profile training performance. Removes itself after training is over."
    order,remove_on_fetch = Recorder.order-1,True
    def __init__(self):
        self._log_full = True
        self._phase, self._train, self._valid = _phase, _train_full, _valid_full

    def before_fit(self):
        self.profiler = self.learn.simple_profiler
        self._start_logging = self.profiler._rolling_average + self.profiler._drop
        self.has_logger = self.profiler.has_logger
        self._start_train_logging, self._start_valid_logging = False, False
        self.n_train_batches = len(self.dls.train)
        self.n_valid_batches = len(self.dls.valid)

    def after_pred(self):
        if self.training:
            self.profiler._raw_values['train_forward'].append(time.perf_counter() - self.profiler._train_batch_start)
            self.profiler._train_loss_start = time.perf_counter()
        else:
            self.profiler._raw_values['valid_predict'].append(time.perf_counter() - self.profiler._valid_batch_start)
            self.profiler._valid_loss_start = time.perf_counter()

    def after_loss(self):
        if self.training:
            self.profiler._raw_values['train_loss'].append(time.perf_counter() - self.profiler._train_loss_start)
        else:
            self.profiler._raw_values['valid_loss'].append(time.perf_counter() - self.profiler._valid_loss_start)

    def after_step(self):
        self.profiler._raw_values['train_opt_step'].append(time.perf_counter() - self.profiler._step_start)
        self.profiler._zero_start = time.perf_counter()

    def after_fit(self):
        self._after_fit([SimpleProfilerCallback, SimpleProfilerPostCallback])

# %% ../../nbs/callback.profiler.ipynb 28
class ProfileMode(str, Enum):
    "Profile enum for `Learner.profile`"
    Throughput = 'throughput'
    Simple     = 'simple'

# %% ../../nbs/callback.profiler.ipynb 29
@patch
def profile(self:Learner,
        mode:ProfileMode=ProfileMode.Throughput, # Which profiler to use. Throughput or Simple.
        show_report:bool=True, # Display formatted report post profile
        plain:bool=False, # For Jupyter Notebooks, display plain report
        markdown:bool=False, # Display markdown formatted report
        save_csv:bool=False,  # Save raw results to csv
        csv_name:str='simpleprofiler.csv', # CSV save location
        rolling_average:int=10, # Number of batches to average throughput over
        drop_first_batch:bool=True, # Drop the first batch from profiling
        logger_callback='wandb' # Log report and samples/second to `logger_callback` using `Callback.name`
    ):
    "Run a fastxtend profiler which removes itself when finished training."
    if mode == ProfileMode.Throughput:
        self.add_cbs([ThroughputCallback(show_report=show_report, plain=plain, markdown=markdown, 
                                         save_csv=save_csv, csv_name=csv_name, rolling_average=rolling_average, 
                                         drop_first_batch=drop_first_batch, logger_callback=logger_callback),
                      ThroughputPostCallback()
                ])
    if mode == ProfileMode.Simple:
        self.add_cbs([SimpleProfilerCallback(show_report=show_report, plain=plain, markdown=markdown, 
                                             save_csv=save_csv, csv_name=csv_name, rolling_average=rolling_average, 
                                             drop_first_batch=drop_first_batch, logger_callback=logger_callback),
                      SimpleProfilerPostCallback()
                ])
    return self

# %% ../../nbs/callback.profiler.ipynb 42
def convert_to_int(s):
    try:
        return int(s.replace(",", ""))
    except ValueError:
        return s

# %% ../../nbs/callback.profiler.ipynb 43
try:
    import wandb

    @patch
    def _wandb_log_after_batch(self:ThroughputCallback, actions:list[str]):
        bs = np.mean(self._raw_values[f'train_bs'][-self._rolling_average:])
        logs = {f'throughput/{action}': self._samples_per_second(bs, action) for action in actions}
        wandb.log(logs, self.learn.wandb._wandb_step+1)

    @patch
    def _wandb_log_after_fit(self:ThroughputCallback):
        for t in self.learn.profile_results.itertuples():
            if isinstance(convert_to_int(t._6), int):
                wandb.summary[f'{t.Phase}/{t.Action}_throughput'] = self._processed_samples[f'{t.Phase}_{t.Action}']
            
            if t.Phase=='fit':
                values = self._raw_values[f'{t.Phase}']
                log = f'{t.Phase}/duration' if t.Action == 'fit' else f'{t.Phase}/{t.Action}_duration'
            else:
                # Optionally drop first batch if train/valid phase
                values = self._raw_values[f'{t.Phase}_{t.Action}'][self._drop:]
                log = f'{t.Phase}/{t.Action}_duration'
            wandb.summary[log] = np.array(values)

        report  = wandb.Table(dataframe=self.learn.profile_report)
        results = wandb.Table(dataframe=self.learn.profile_results)

        wandb.log({"profile_report": report})
        wandb.log({"profile_results": results})
except:
    pass
