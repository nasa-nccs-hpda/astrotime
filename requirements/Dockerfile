# Arguments to pass to the image
ARG VERSION_DATE=24.09
ARG FROM_IMAGE=nvcr.io/nvidia/pytorch

# Import RAPIDS container as the BASE Image (cuda base image)
FROM ${FROM_IMAGE}:${VERSION_DATE}-py3

# Ubuntu needs noninteractive to be forced
ENV DEBIAN_FRONTEND noninteractive

# System dependencies
RUN apt-get update && \
    apt-get -y install software-properties-common && \
    apt-get update && apt-get -y dist-upgrade && \
    apt-get -y install build-essential \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libfontconfig1 \
        bzip2 \
        diffutils \
        file \
        build-essential \
        make \
        swig \
        libsqlite3-dev \
        libx11-dev \
        libproj-dev \
        proj-data \
        proj-bin \
        wget \
        vim \
        curl \
        git \
        procps \
        gcc \
        g++ \
        bzip2 \
        libbz2-dev \
        libssl-dev \
        libzmq3-dev \
        libpng-dev \
        libfreetype6-dev \ 
        locales && \
    apt-get -y autoremove && \
    rm -rf /var/cache/apt /var/lib/apt/lists/*

# Pip
RUN pip --no-cache-dir install \
    transformers \
    datasets \
    webdataset \
    deepspeed \
    'huggingface_hub[cli,torch]' \
    scipy \
    astropy \
    lightkurve \
    fitsio \
    scikit-learn \
    pandas \
    numpy \
    tqdm \
    webcolors \
    pytest \
    coveralls \
    rtree \
    diskcache \
    ninja \
    xarray \
    netCDF4 \
    torchcde \
    torchdiffeq \
    hydra-core \
    rich \
    timm \
    supervision \
    pytest \
    coveralls \
    git+https://github.com/nasa-nccs-hpda/astrotime.git@main

HEALTHCHECK NONE
ENTRYPOINT []
CMD ["/bin/bash"]
