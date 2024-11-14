# hash:sha256:371bf441e56d9d9ffcfd52a9973e12488c057e385eb66336659d37be6fbbac67
FROM registry.codeocean.allenneuraldynamics.org/codeocean/pytorch-tensorflow-jax:2.1

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

ARG new_pred_version="1.0.0"

RUN pip install --no-cache-dir \
    jupyterlab \
    git+https://github.com/AllenNeuralDynamics/aind-large-scale-prediction.git \
    numpy==1.24.4 \
    scikit-image==0.22.0 \
    matplotlib==3.8.2 \
    scipy==1.10.1 \
    cupy-cuda11x \
    neuroglancer