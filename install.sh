conda create -n fastxtend python=3.10 "pytorch>=2.0.1" torchvision torchaudio \
pytorch-cuda=11.8 cuda fastai nbdev pkg-config libjpeg-turbo opencv tqdm \
terminaltables psutil numpy numba librosa=0.9.2 timm kornia rich typer \
jupyterlab ipywidgets wandb \
-c pytorch -c nvidia/label/cuda-11.8.0 -c fastai -c huggingface -c conda-forge