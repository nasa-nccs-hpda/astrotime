# Arguments to pass to the image
ARG VERSION_DATE=25.04
ARG FROM_IMAGE=nvcr.io/nvidia/pytorch
ARG ASTROTIME_TAG=0.3.0

# Import RAPIDS container as the BASE Image (cuda base image)
FROM ${FROM_IMAGE}:${VERSION_DATE}-py3

# Pass ARGs again after FROM (Docker limitation)
ARG ASTROTIME_TAG

# Ubuntu needs noninteractive to be forced
ENV DEBIAN_FRONTEND=noninteractive

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
        libbz2-dev \
        libssl-dev \
        libzmq3-dev \
        libpng-dev \
        libfreetype6-dev \
        locales && \
    apt-get -y autoremove && \
    rm -rf /var/cache/apt /var/lib/apt/lists/*

# Python dependencies
RUN pip --no-cache-dir install \
    transformers \
    "numpy<2" \
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
    git+https://github.com/nasa-nccs-hpda/astrotime.git@${ASTROTIME_TAG}

# Clone astrotime for access to scripts
RUN git clone --branch ${ASTROTIME_TAG} https://github.com/nasa-nccs-hpda/astrotime.git /usr/local/ilab/astrotime

# Entrypoint
HEALTHCHECK NONE
ENTRYPOINT []
CMD ["/bin/bash"]
