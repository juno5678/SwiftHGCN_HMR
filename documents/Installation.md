## Installation

Our codebase is developed based on Ubuntu 22.04 and Pytorch framework.

### Requirements

* Python >=3.9
* Pytorch >=2.1.1
* torchvision >= 0.16.1
* CUDA >= 11.8
* cuDNN (if CUDA available)

### Installation with conda

# Install SwiftHgcnHMR
git clone --recursive git@github.com:juno5678/SwiftHGCN_HMR.git

```bash
# We suggest to create a new conda environment with python version 3.9
conda create --name MambaHMR python=3.9

# Activate conda environment
conda activate MambaHMR

# Install Pytorch that is compatible with your CUDA version
# CUDA 11.8
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118

# Install pkg
pip install packaging
cd pkg
pip install causal_conv1d==1.1.0
pip install -e mamba-1p1p1
pip install ./manopth/.

# Install apex
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install 
cd ..

# Install OpenDR
pip install matplotlib
#pip install git+https://gitlab.eecs.umich.edu/ngv-python-modules/opendr.git
cd opendr
python setup.py build
python setup.py install

# Install causal_conv1d
pip install causal_conv1d==1.1.0

# Install mamba
pip install -e ./pkg/mamba-1p1p1




# Install requirements
pip install -r requirements.txt
pip install ./pkg/manopth/.
pip install --upgrade azureml-core


```
