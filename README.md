# CrossLink

## Description
* A simple self-supervised learning method for cross-modal pretraining.

## Requirements
* Linux with Python 3.8
* Conda

## Setup
- Setup conda environment:
```bash
# Create environment
conda create -n cim python=3.8 -y
conda activate cim

# Install requirements
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch -y
```

- Navigate to the root directory of this repository:
```bash
cd CrossLink

# Install other requirements
pip install -r requirements.txt
```

## Pretrain

To pre-train `<method = any(mim, dino, cim)>` using `ViT-Small` as the backbone, run the following on GPUs with port 8888:
```shell
sh scripts/dist_pretrain.sh 1 8888 <path-to-imagenet> <method> small none <job-name>
```
