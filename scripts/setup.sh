#!/bin/bash

#conda create -n ShgcnHMR python=3.9

#. /home/juno/anaconda3/bin/activate ShgcnHMR
#conda activate SHgcnHMR

pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118


# Install pkg
pip install packaging
cd pkg
pip install causal_conv1d==1.1.0
pip install -e mamba-1p1p1
pip install ./manopth/.


git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install --cuda_ext --cpp_ext
cd ../

### Install OpenDR
pip install matplotlib
#pip install git+https://gitlab.eecs.umich.edu/ngv-python-modules/opendr.git
cd opendr
python setup.py build
python setup.py install
### Install requirements
cd ../..
pip install -r requirements.txt
pip install --upgrade azureml-core
