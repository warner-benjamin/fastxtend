fastxtend examples
==================

This folder contains example scripts for using fastxtend.

imagenette.py
-------------

`imagenette.py` allows training on [Imagenette and
ImageWoof](https://github.com/fastai/imagenette) with most of fastxtend
features as options. It requires [Typer](https://typer.tiangolo.com).

Run `python imagenette.py train -h` to get a full list of training
options. And `python imagenette.py create -h` for FFCV dataset creation
options.

`imagenette.yaml` is an optional config file for changing the
`imagenette.py train` defaults. Load the config file by passing
`train --config imagenette.yaml` to `imagenette.py`. Passed CLI options
to `train` will override any config file settings.

To recreate the fastxtend example bencmark on your own system, run:

```bash
# warmup & dataset creation
python imagenette.py train --epochs 1
python imagenette.py train --fastai --standard --full-size --epochs 1

# fastai
python imagenette.py train --fastai --standard --full-size

# progressive resizing & fused optimizer with fastai dataloader
python imagenette.py train --fastai

# progressive resizing & fused optimizer with ffcv dataloader
python imagenette.py train
```