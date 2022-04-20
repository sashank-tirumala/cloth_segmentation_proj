# Cloth Segmentation Network Training Code

This directory contains code for training the segmentation network. 

# Installation
`pip install -r requirements.txt`

# Training
0. Download the [dataset](https://drive.google.com/drive/folders/18Qr5umjP71jNGQh6eM5ck3u1Aui6I4Sd?usp=sharing)
1. Edit configs in `configs/segmentation.json`
2. `python train.py`

# Setting up on the cluster
0. `salloc -p GPU --mem=12G -t 0 -w compute-0-7`
1. `ssh compute-0-7` Change the compute number as required
2. `cd /scratch/sashank/11785_project`
3. ` rsync -av sashank@128.2.176.255:/media/YertleDrive4/layer_grasp/dataset ./`
4. Now the datasets are synced up. We need to sync up the code