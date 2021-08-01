FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

# Fix DL4006
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# ---- Miniforge installer ----
# Default values can be overridden at build time
# Check https://github.com/conda-forge/miniforge/releases
# Conda version
ARG CONDA_VERSION="4.9.2"
# Miniforge installer patch version
ARG MINIFORGE_PATCH_NUMBER="7"
# Package Manager and Python implementation to use (https://github.com/conda-forge/miniforge)
# - conda only: either Miniforge3 to use Python or Miniforge-pypy3 to use PyPy
# - conda + mamba: either Mambaforge to use Python or Mambaforge-pypy3 to use PyPy
ARG MINIFORGE_PYTHON="Mambaforge"

# Miniforge archive to install
ARG MINIFORGE_VERSION="${CONDA_VERSION}-${MINIFORGE_PATCH_NUMBER}"
# Miniforge installer
ARG MINIFORGE_INSTALLER="${MINIFORGE_PYTHON}-${MINIFORGE_VERSION}-Linux-x86_64.sh"
# Miniforge checksum
ARG MINIFORGE_CHECKSUM="5a827a62d98ba2217796a9dc7673380257ed7c161017565fba8ce785fb21a599"

# Python version. Default value is the same as miniforge default version
ARG PYTHON_VERSION=default
# Force debian to accept default values for commands
ARG DEBIAN_FRONTEND=noninteractive

# Install basics
RUN set -xe && \
    apt-get -qq update && \
    apt-get install -yqq --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    fonts-liberation \
    gcc \
    git \
    libssl-dev \
    locales \
    lsof \
    tar \
    unzip \
    wget \ 
    zip && \
    apt-get clean && \
    apt-get autoremove

RUN set -xe && \
    # Force locale to en-US
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen && \
    # Enable prompt color in the skeleton .bashrc before creating the non-root users
    # hadolint ignore=SC2016
    sed -i 's/^#force_color_prompt=yes/force_color_prompt=yes/' /etc/skel/.bashrc && \
    # Add call to conda init script see https://stackoverflow.com/a/58081608/4413446
    echo 'eval "$(command conda shell.bash hook 2> /dev/null)"' | tee -a ~/.bashrc >> /etc/skel/.bashrc

# Install miniforge
# Prerequisites installation: conda, mamba, pip, tini
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

RUN set -xe && \
    wget --quiet "https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/${MINIFORGE_INSTALLER}" && \
    echo "${MINIFORGE_CHECKSUM} *${MINIFORGE_INSTALLER}" | sha256sum --check && \
    /bin/bash "${MINIFORGE_INSTALLER}" -f -b -p $CONDA_DIR && \
    rm "${MINIFORGE_INSTALLER}" && \
    # Conda configuration see https://conda.io/projects/conda/en/latest/configuration.html
    echo "conda ${CONDA_VERSION}" >> $CONDA_DIR/conda-meta/pinned && \
    conda config --system --set auto_update_conda false && \
    conda config --system --set show_channel_urls true && \
    if [ ! $PYTHON_VERSION = 'default' ]; then \
        conda install --yes python=$PYTHON_VERSION; \
    fi && \
    # Pin python version only to update only maintenance releases
    conda list python | grep '^python ' | tr -s ' ' | cut -d '.' -f 1,2 | sed 's/$/.*/' >> $CONDA_DIR/conda-meta/pinned && \
    conda install --quiet --yes \
        conda=${CONDA_VERSION} \
        pip \
        tini=0.18.0 && \
    conda update --all --quiet --yes && \
    conda list tini | grep tini | tr -s ' ' | cut -d ' ' -f 1,2 >> $CONDA_DIR/conda-meta/pinned && \
    conda clean --all -f -y

# Fix locales once again, set heptabot to run on GPU
RUN apt-get -qq update --fix-missing && apt-get -qq install locales
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LC_ALL en_US.UTF-8
ENV MODEL_PLACE gpu

# Copy repo files to working directory and mark it as such
COPY . /root
WORKDIR /root

# Initialise virtual environment with python 3.6.9
RUN mamba create -y -q -n heptabot python=3.6.9

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "heptabot", "/bin/bash", "-c"]

# Install requirements
RUN mamba install -yq -c conda-forge --file conda_requirements.txt
RUN pip install -q -r requirements.txt
RUN pip install -q transformers==4.1.1
RUN mamba install -yq jupyterlab

# Set up nltk and spaCy
RUN python -c "import nltk; nltk.download(\"punkt\")"
RUN python -m spacy download -d en_core_web_sm-1.2.0 1>/dev/null 2>&1
RUN python -m spacy link en_core_web_sm en
RUN pip install -q --upgrade pip

# Download models
RUN mkdir models
RUN wget -q https://storage.googleapis.com/heptabot/models/external/distilbert_stsb_model.tar.gz -P ./models
RUN tar -xzf ./models/distilbert_stsb_model.tar.gz -C ./models
RUN rm ./models/distilbert_stsb_model.tar.gz
RUN mkdir ./models/classifier
RUN wget -q https://storage.googleapis.com/heptabot/models/classifier/err_type_classifier.cbm -P ./models/classifier
RUN mkdir ./models/t5-tokenizer
RUN wget -q https://storage.googleapis.com/heptabot/models/external/sentencepiece.model -P ./models/t5-tokenizer
RUN mv ./models/t5-tokenizer/sentencepiece.model ./models/t5-tokenizer/spiece.model
RUN wget -q https://storage.googleapis.com/heptabot/models/external/tokenizer.json -P ./models/t5-tokenizer
RUN mkdir ./models/savemodel
RUN wget -q https://storage.googleapis.com/heptabot/models/medium/gpu/saved_model.pb -P ./models/savemodel
RUN mkdir ./models/savemodel/variables
RUN wget -q https://storage.googleapis.com/heptabot/models/medium/gpu/variables/variables.data-00000-of-00002 -P ./models/savemodel/variables
RUN wget -q https://storage.googleapis.com/heptabot/models/medium/gpu/variables/variables.data-00001-of-00002 -P ./models/savemodel/variables
RUN wget -q https://storage.googleapis.com/heptabot/models/medium/gpu/variables/variables.index -P ./models/savemodel/variables
RUN chmod +x start.sh

# Set Flask for serving
EXPOSE 5000

# Move the Docker notebook to main directory
RUN mv ./notebooks/Run_medium_model_from_Docker_image.ipynb .
RUN rm -rf notebooks

# Remove redundant lists
RUN rm -rf /var/lib/apt/lists/* /tmp/*

SHELL ["/bin/sh", "-c"]
ENTRYPOINT ["/bin/bash", "-l", "-c"]
