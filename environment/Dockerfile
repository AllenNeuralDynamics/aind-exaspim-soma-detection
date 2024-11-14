# hash:sha256:5591d4b68169c2c60e53e8a229b617c41ebffd4d99d4ec889e87d68a81767798
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/miniconda3:4.9.2-cuda11.7.0-cudnn8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    matplotlib \
    s3fs \
     cupy-cuda11x

RUN conda install -y \
        jupyterlab==4.0.11 \
    && conda clean -ya

COPY postInstall /
RUN /postInstall
