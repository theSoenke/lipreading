FROM nvidia/cuda:10.1-devel

RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    ca-certificates \
    libjpeg-dev \
    ffmpeg \
    cmake \
    libsm6 \
    git

RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
    /opt/conda/bin/conda install conda-build
ENV PATH=$PATH:/opt/conda/bin/

WORKDIR /project
COPY environment.yml .
RUN conda env create -f environment.yml && \
    conda clean -afy

# Required for beamsearch
# RUN git clone --recursive https://github.com/parlance/ctcdecode.git && \
#     cd ctcdecode && conda run -n lipreading pip install .
RUN git clone https://github.com/NVIDIA/apex && cd apex && \
    conda run -n lipreading pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

COPY . .

ENTRYPOINT ["/bin/bash", "scripts/docker/run.sh"]
