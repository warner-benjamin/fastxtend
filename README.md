# fastxtend
> fastxtend (fastai extended) is a collection of tools, extensions, and addons for fastai


## Documentation
https://fastxtend.benjaminwarner.dev

## Feature overview

**General Features**

* Flexible metrics which can log on train, valid, or both. Backwards compatible with fastai metrics.
* Easily use multiple losses and log each individual loss on train and valid.
* A simple profiler for profiling fastai training.

**Vision**

* Increase training speed using `ProgressiveResize` to automaticly apply progressive resizing.
* Apply `MixUp`, `CutMix`, or Augmentations with `CutMixUp` or `CutMixUpAugment`.
* Additional image augmentations
* Support for running fastai batch transforms on CPU.
* More attention modules
* A flexible implementation of fastai’s xresnet.

**Audio**

* `TensorAudio`, `TensorSpec`, `TensorMelSpec` objects which maintain metadata and support plotting themselves using librosa.
* A selection of performant audio augmentations inspired by fastaudio and torch-audiomentations.
* Uses TorchAudio to quickly convert `TensorAudio` waveforms into `TensorSpec` spectrograms or `TensorMelSpec` mel spectrograms using the GPU.
* Out of the box support for converting one `TensorAudio` to one or multiple `TensorSpec` or `TensorMelSpec` objects from the Datablock api.
* Audio MixUp and CutMix Callbacks.
* `audio_learner` which merges multiple `TensorSpec` or `TensorMelSpec` objects before passing to the model.

Check out the documentation for additional splitters, callbacks, schedulers, utilities, and more.

## Install

To install, run:
```
pip install fastxtend
```

To install with dependencies for vision, audio, or all tasks run one of:
```
pip install fastxtend[vision]

pip install fastxtend[audio]

pip install fastxtend[all]
```

Or to create an editable install:
```
git clone https://github.com/warner-benjamin/fastxtend.git
cd fastxtend
pip install -e ".[dev]"
```

## Usage
Like fastai, fastxtend provides safe wildcard imports using python’s `__all__`. 
```python
from fastai.vision.all import *
from fastxtend.vision.all import *
```
In general, import fastxtend after all fastai imports, as fastxtend modifies fastai. Any method modified by fastxtend is backwards compatible with the original fastai code.

## Examples
Log an accuracy metric on the training set as a smoothed metric and validation set like normal:
```python
Learner(..., metrics=[Accuracy(log_metric=LogMetric.Train, metric_type=MetricType.Smooth),
                      Accuracy()])
```

Log multiple losses as individual metrics on train and valid:
```python
mloss = MultiLoss(loss_funcs=[nn.MSELoss, nn.L1Loss], 
                  weights=[1, 3.5], loss_names=['mse_loss', 'l1_loss'])

Learner(..., loss_func=mloss, metrics=RMSE(), cbs=MultiLossCallback)
```

Apply MixUp, CutMix, or Augmentation while training:
```python
Learner(..., cbs=CutMixUpAugment)
```

Profile a fastai training loop:
```python
from fastxtend.callback import simpleprofiler

learn = Learner(...).profile()
learn.fit_one_cycle(2, 3e-3)
```

Train in channels last format:
```python
Learner(...).to_channelslast()
```

## Requirements

fastxtend requires fastai to be installed. See http://docs.fast.ai/ for installation instructions.
