FROM continuumio/anaconda3
#FROM continuumio/anaconda3:2020.02

ENV PYTHONUNBUFFERED=1

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    curl \
    ca-certificates \
    git \
    zip \
    vim \
    cmake pkg-config gfortran \
    && \
    apt-get clean && rm -rf /var/cache/apt/archives/* /var/lib/apt/lists/* # clean up

#RUN mkdir /opt/conda/pkgs
#RUN chown 1000:1000 /opt/conda

# Create non-root user
ARG NB_USER=vscode
ARG NB_UID=1000
RUN useradd -u $NB_UID -m $NB_USER -s /bin/bash && \
    echo 'vscode ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER $NB_USER
ENV PATH "/home/${NB_USER}/.local/bin:${PATH}"
ENV PYTHONPATH "/home/${NB_USER}/work/src:${PYTONPATH}"

# for vscode
RUN mkdir /home/${NB_USER}/work

RUN conda config --add pkgs_dirs /home/vscode/.conda/pkgs