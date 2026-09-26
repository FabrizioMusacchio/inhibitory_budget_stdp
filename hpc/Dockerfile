FROM continuumio/miniconda3:23.10.0-1

SHELL ["/bin/bash", "-lc"]

# Keep the image CPU-only and small. TensorFlow is intentionally omitted because
# the cluster run scripts use a copied MNIST mnist.npz file directly.
RUN conda config --set ssl_verify False && \
    conda config --remove channels defaults && \
    conda config --add channels conda-forge && \
    conda config --set channel_priority strict

RUN conda create -n diehl_cook_euler -y python=3.12 mamba && \
    conda run -n diehl_cook_euler mamba install -y \
      numpy \
      matplotlib \
      pandas \
      numba \
      scikit-learn && \
    conda clean -afy

ENV CONDA_DEFAULT_ENV=diehl_cook_euler
ENV CONDA_PREFIX=/opt/conda/envs/diehl_cook_euler
ENV PATH=/opt/conda/envs/diehl_cook_euler/bin:/opt/conda/bin:$PATH
ENV MPLBACKEND=Agg
ENV PYTHONUNBUFFERED=1

RUN /opt/conda/envs/diehl_cook_euler/bin/python -c "import numpy, matplotlib, numba, sklearn; print('diehl_cook_euler container ok')" && \
    conda run -n diehl_cook_euler python -c "import numpy, matplotlib, numba, sklearn; print('conda run ok')"

CMD ["/bin/bash"]
