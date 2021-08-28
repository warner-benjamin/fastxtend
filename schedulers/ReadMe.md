# fit_flat_varied

Fit `self.model` for `n_epoch` at flat `start_lr`, then change to flat `next_lr` at `change_by`, optionally with cosine annealing over `change_time`. Final cosine annealing at `pct_start`.

Args:

`n_epoch`, `start_lr`, `div_final`, `pct_start`, `wd`, `cbs`, & `reset_opt` are all same as [`fit_flat_cos`](https://docs.fast.ai/callback.schedule.html#Learner.fit_flat_cos) from fast.ai.

`next_lr`: single or list of learning rates to switch to at change_by. Must be same length as change_by.

`change_by`: single or list of epochs or percent of training to switch to `next_lr `by. Must be same length as `next_lr`.

`change_time`: if greater than 0 (pct of training or epochs), how long to cosine anneal to `next_lr`. Can single or list. 



Example Schedules:

`learn.fit_flat_varied(15, 8e-3, next_lr=[7e-3, 5e-3], change_by=[4,8], change_time=2)`

![cosine_annealing](fit_flat_varied_1.png)



`learn.fit_flat_varied(10, 8e-3, change_by=[0.2, 0.4, 0.6], next_lr=[7e-3, 9e-3, 6e-3], change_time=0)`

![immediate change](fit_flat_varied_2.png)

