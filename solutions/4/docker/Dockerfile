FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04
RUN echo "/usr/local/cuda/lib" >> /etc/ld.so.conf.d/cuda.conf && \
    echo "/usr/local/cuda/lib64" >> /etc/ld.so.conf.d/cuda.conf && \
    echo "/usr/local/cuda/compat" >> /etc/ld.so.conf.d/cuda.conf
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/targets/x86_64-linux/lib/stubs/:/usr/local/cuda/lib:/usr/local/cuda/lib64:/usr/local/cuda/compat/
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git curl vim ca-certificates libjpeg-dev libpng-dev openbabel && \
    rm -rf /var/lib/apt/lists/*
RUN curl -o ~/miniconda.sh -O https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh
ENV PATH=/opt/conda/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
RUN conda update -y conda && \
    conda config --add channels bioconda && \
    conda config --add channels conda-forge && \
    conda config --add channels pytorch && \
    conda config --add channels dglteam && \
    conda config --add channels fastai && \
    conda config --set channel_priority strict && \
    conda update --all -y && \
    conda install -y python=$PYTHON_VERSION \
        tornado numpy pyyaml scipy ipython mkl mkl-include cython typing \
        pandas pytables cytoolz rdkit jupyterlab humanfriendly rdflib path.py future \
        feather-format networkx google-api-python-client boto3 \
        bottle py-cpuinfo requests \
        matplotlib seaborn plotly \
        h5py google-pasta wrapt \
        nodejs ipywidgets widgetsnbextension jupyter_contrib_nbextensions tqdm \
        pytorch-nightly torchvision cudatoolkit=10.0 \
        bx-python pysam cyvcf2 pybedtools \
        nvidia-ml-py3 \
        dgl-cuda10.0 && \
    conda clean -ya
RUN pip install --no-cache-dir \
    adabound duecredit tb-nightly jupyter-tensorboard \
    glances[action,browser,cpuinfo,gpu,ip,web] && \
    pip install --no-cache-dir tf-nightly
RUN conda install -y mdanalysis mock mdtraj simpletraj && \
    conda clean -ya && \
    pip --no-cache-dir install nglview
RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager && \
    jupyter-labextension install nglview-js-widgets && \
    jupyter-labextension enable nglview
RUN ipython profile create --reset default && \
    jupyter notebook --generate-config
WORKDIR /workspace
RUN chmod -R a+w /workspace
CMD [ "/bin/bash", "-c", "jupyter lab --allow-root --ip=0.0.0.0 --port=8888 --notebook-dir=/workspace/ --NotebookApp.token=''"]

