# hash:sha256:4cf12196320a9ec3c377025e37e0eb3dcd41afb9a3d5d463a482ed90578df58d
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/miniconda3:4.9.2-cuda11.7.0-cudnn8-ubuntu20.04

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    black \
    dominate \
    elasticdeform \
    flake8 \
    imagej \
    matplotlib \
    opencv-python \
    scikit-image \
    s3fs \
    tensorboard \
    zarr

RUN pip install --no-cache-dir \
    torch==1.10.0+cu113 \
    torchvision==0.11.1+cu113 \
    torchaudio==0.10.2+cu113 \
    -f https://download.pytorch.org/whl/cu113/torch_stable.html

RUN conda install -y \
        jupyterlab==4.0.11 \
    && conda clean -ya
