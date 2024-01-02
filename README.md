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

## Pretrain

To pre-train `<method = any(mim, dino, cim)>` using `ViT-Small` as the backbone, run the following on 8 A100 GPUs with port 8888:
```shell
sh scripts/dist_pretrain.sh 8 8888 <path-to-imagenet> <method> small none <job-name>
```
