conda create -n fastxtend python=3.10 pytorch torchvision torchaudio pytorch-cuda=11.8 \
cuda fastai nbdev pkg-config libjpeg-turbo opencv tqdm terminaltables psutil \
numpy=1.23.5 numba librosa=0.9.2 timm kornia rich typer jupyterlab ipykernel wandb \
-c pytorch -c nvidia/label/cuda-11.8.0 -c fastai -c huggingface -c conda-forge