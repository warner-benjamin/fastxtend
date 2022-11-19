fastxtend
================

fastxtend (fastai extended) is a collection of tools, extensions, and
addons for fastai

## Feature overview

**General Features**

- [Fused optimizers](optimizer.fused.html) which are 21 to 293 percent
  faster relative to fastai native optimizers.
- Flexible [metrics](metrics.html) which can log on train, valid, or
  both. Backwards compatible with fastai metrics.
- Easily use [multiple losses](multiloss.html) and log each individual
  loss on train and valid.
- A [simple profiler](callback.simpleprofiler.html) for profiling fastai
  training.

**Vision**

- Increase training speed using
  [`ProgressiveResize`](https://fastxtend.benjaminwarner.dev/callback.progresize.html#progressiveresize)
  to automaticly apply progressive resizing.
- Apply
  [`MixUp`](https://fastxtend.benjaminwarner.dev/callback.cutmixup.html#mixup),
  [`CutMix`](https://fastxtend.benjaminwarner.dev/callback.cutmixup.html#cutmix),
  or Augmentations with
  [`CutMixUp`](https://fastxtend.benjaminwarner.dev/callback.cutmixup.html#cutmixup)
  or
  [`CutMixUpAugment`](https://fastxtend.benjaminwarner.dev/callback.cutmixup.html#cutmixupaugment).
- Additional [image augmentations](vision.augment.batch.html).
- Support for running fastai [batch transforms on
  CPU](vision.data.html).
- More [attention](vision.models.attention_modules.html) and
  [pooling](vision.models.pooling.html) modules
- A flexible implementation of fastai’s
  [`XResNet`](https://fastxtend.benjaminwarner.dev/vision.models.xresnet.html#xresnet).

**Audio**

- [`TensorAudio`](https://fastxtend.benjaminwarner.dev/audio.core.html#tensoraudio),
  [`TensorSpec`](https://fastxtend.benjaminwarner.dev/audio.core.html#tensorspec),
  [`TensorMelSpec`](https://fastxtend.benjaminwarner.dev/audio.core.html#tensormelspec)
  objects which maintain metadata and support plotting themselves using
  librosa.
- A selection of performant [audio augmentations](audio.augment.html)
  inspired by fastaudio and torch-audiomentations.
- Uses TorchAudio to quickly convert
  [`TensorAudio`](https://fastxtend.benjaminwarner.dev/audio.core.html#tensoraudio)
  waveforms into
  [`TensorSpec`](https://fastxtend.benjaminwarner.dev/audio.core.html#tensorspec)
  spectrograms or
  [`TensorMelSpec`](https://fastxtend.benjaminwarner.dev/audio.core.html#tensormelspec)
  mel spectrograms using the GPU.
- Out of the box support for converting one
  [`TensorAudio`](https://fastxtend.benjaminwarner.dev/audio.core.html#tensoraudio)
  to one or multiple
  [`TensorSpec`](https://fastxtend.benjaminwarner.dev/audio.core.html#tensorspec)
  or
  [`TensorMelSpec`](https://fastxtend.benjaminwarner.dev/audio.core.html#tensormelspec)
  objects from the Datablock api.
- Audio [MixUp and CutMix](audio.mixup.html) Callbacks.
- [`audio_learner`](https://fastxtend.benjaminwarner.dev/audio.learner.html#audio_learner)
  which merges multiple
  [`TensorSpec`](https://fastxtend.benjaminwarner.dev/audio.core.html#tensorspec)
  or
  [`TensorMelSpec`](https://fastxtend.benjaminwarner.dev/audio.core.html#tensormelspec)
  objects before passing to the model.

Check out the documentation for additional splitters, callbacks,
schedulers, utilities, and more.

## Documentation

<https://fastxtend.benjaminwarner.dev>

## Install

fastxtend is avalible on pypi:

``` default
pip install fastxtend
```

To install with dependencies for vision, audio, or all tasks run one of:

``` default
pip install fastxtend[vision]

pip install fastxtend[audio]

pip install fastxtend[all]
```

Or to create an editable install:

``` default
git clone https://github.com/warner-benjamin/fastxtend.git
cd fastxtend
pip install -e ".[dev]"
```

## Usage

Like fastai, fastxtend provides safe wildcard imports using python’s
`__all__`.

``` python
from fastai.vision.all import *
from fastxtend.vision.all import *
```

In general, import fastxtend after all fastai imports, as fastxtend
modifies fastai. Any method modified by fastxtend is backwards
compatible with the original fastai code.

## Examples

Use a fused ForEach optimizer:

``` python
Learner(..., opt_func=adam(fused=True))
```

Log an accuracy metric on the training set as a smoothed metric and
validation set like normal:

``` python
Learner(..., metrics=[Accuracy(log_metric=LogMetric.Train, metric_type=MetricType.Smooth),
                      Accuracy()])
```

Log multiple losses as individual metrics on train and valid:

``` python
mloss = MultiLoss(loss_funcs=[nn.MSELoss, nn.L1Loss],
                  weights=[1, 3.5], loss_names=['mse_loss', 'l1_loss'])

Learner(..., loss_func=mloss, metrics=RMSE(), cbs=MultiLossCallback)
```

Apply MixUp, CutMix, or Augmentation while training:

``` python
Learner(..., cbs=CutMixUpAugment)
```

Profile a fastai training loop:

``` python
from fastxtend.callback import simpleprofiler

learn = Learner(...).profile()
learn.fit_one_cycle(2, 3e-3)
```

Train in channels last format:

``` python
Learner(...).to_channelslast()
```

## Requirements

fastxtend requires fastai to be installed. See <http://docs.fast.ai> for
installation instructions.
