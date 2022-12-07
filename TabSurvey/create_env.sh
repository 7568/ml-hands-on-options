# Use anaconda as baseline
FROM /home/liyu/miniconda3

# Install Jupyter notebook
# /home/liyu/miniconda3/bin/conda install jupyter -y
# RUN mkdir /opt/notebooks

# This is just to avoid the token all the time!
# RUN opt/conda/bin/jupyter notebook --generate-config
# COPY jupyter_notebook_config.json root/.jupyter
# Use Password "1234"
conda env remove -n sklearn
conda env remove -n gbdt
conda env remove -n torch
conda env remove -n tensorflow
#############################################################################################################

# Set up Sklearn environment
/home/liyu/miniconda3/bin/conda create -n sklearn -y python=3.8 scikit-learn
/home/liyu/miniconda3/bin/conda install -n sklearn -y -c anaconda ipykernel
/home/liyu/miniconda3/envs/sklearn/bin/python -m ipykernel install --user --name=sklearn
/home/liyu/miniconda3/bin/conda install -n sklearn -y -c conda-forge optuna
/home/liyu/miniconda3/bin/conda install -n sklearn -y -c conda-forge configargparse
/home/liyu/miniconda3/bin/conda install -n sklearn -y pandas

#############################################################################################################

# Set up GBDT environment
/home/liyu/miniconda3/bin/conda create -n gbdt -y python=3.8
/home/liyu/miniconda3/bin/conda install -n gbdt -y -c anaconda ipykernel
/home/liyu/miniconda3/envs/gbdt/bin/python -m ipykernel install --user --name=gbdt
/home/liyu/miniconda3/envs/gbdt/bin/python -m pip install xgboost==1.5.0
/home/liyu/miniconda3/envs/gbdt/bin/python -m pip install catboost==1.0.3
/home/liyu/miniconda3/envs/gbdt/bin/python -m pip install lightgbm==3.3.1
/home/liyu/miniconda3/bin/conda install -n gbdt -y -c conda-forge optuna
/home/liyu/miniconda3/bin/conda install -n gbdt -y -c conda-forge configargparse
/home/liyu/miniconda3/bin/conda install -n gbdt -y pandas

# For ModelTrees
/home/liyu/miniconda3/envs/gbdt/bin/python -m pip install https://github.com/schufa-innovationlab/model-trees/archive/master.zip

#############################################################################################################

# Set up Pytorch environment
/home/liyu/miniconda3/bin/conda create -n torch -y python=3.8 pytorch cudatoolkit=11.3 -c pytorch
/home/liyu/miniconda3/bin/conda install -n torch -y -c anaconda ipykernel
/home/liyu/miniconda3/bin/conda install -n torch -y -c conda-forge optuna
/home/liyu/miniconda3/bin/conda install -n torch -y -c conda-forge configargparse
/home/liyu/miniconda3/bin/conda install -n torch -y scikit-learn
/home/liyu/miniconda3/bin/conda install -n torch -y pandas
/home/liyu/miniconda3/bin/conda install -n torch -y matplotlib
/home/liyu/miniconda3/bin/conda install -n torch -y -c pytorch captum
/home/liyu/miniconda3/bin/conda install -n torch -y shap
/home/liyu/miniconda3/envs/gbdt/bin/python -m ipykernel install --user --name=torch

# For TabNet
/home/liyu/miniconda3/envs/torch/bin/python -m pip install pytorch-tabnet

# For NODE
/home/liyu/miniconda3/envs/torch/bin/python -m pip install requests
/home/liyu/miniconda3/envs/torch/bin/python -m pip install qhoptim

# For DeepGBM
/home/liyu/miniconda3/envs/torch/bin/python -m pip install lightgbm==3.3.1

# For TabTransformer
/home/liyu/miniconda3/envs/torch/bin/python -m pip install einops

#############################################################################################################

# Set up Keras environment
/home/liyu/miniconda3/bin/conda create -n tensorflow -y tensorflow-gpu=1.15.0 keras
/home/liyu/miniconda3/bin/conda install -n tensorflow -y -c anaconda ipykernel
/home/liyu/miniconda3/bin/conda install -n tensorflow -y -c conda-forge optuna
/home/liyu/miniconda3/bin/conda install -n tensorflow -y -c conda-forge configargparse
/home/liyu/miniconda3/bin/conda install -n tensorflow -y scikit-learn
/home/liyu/miniconda3/bin/conda install -n tensorflow -y pandas

#############################################################################################################

# For STG
/home/liyu/miniconda3/envs/torch/bin/python -m pip install stg==0.1.2

# For NAM
/home/liyu/miniconda3/envs/torch/bin/python -m pip install https://github.com/AmrMKayid/nam/archive/main.zip
/home/liyu/miniconda3/envs/torch/bin/python -m pip install tabulate

# For DANet
/home/liyu/miniconda3/envs/torch/bin/python -m pip install yacs

#############################################################################################################

# # Download code into container
# RUN git clone https://github.com/kathrinse/TabSurvey.git /opt/notebooks
# # Start jupyter notebook
# CMD opt/conda/bin/jupyter notebook --notebook-dir=/opt/notebooks --ip='*' --port=3123 --no-browser --allow-root