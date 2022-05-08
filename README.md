# Cloth Segmentation Network Training Code

This directory contains code for training the segmentation network. 

# Installation
`pip install -r requirements.txt`

# Training
0. Download the [dataset](https://drive.google.com/drive/folders/18Qr5umjP71jNGQh6eM5ck3u1Aui6I4Sd?usp=sharing)
2. `CUDA_VISIBLE_DEVICES=0 singularity exec --nv /home/stirumal/singularity/pytorch.sif python train.py -lr 1e-3  -wd 0 -m 0.9 -ss 30 -g 0.5 -bs 8 -e 300 -dp /scratch/sashank/11785_project/dataset/test  -rp /scratch/sashank/11785_project/runs -t False -nc 2 -nf 2 `

# Setting up on the cluster
0. `salloc -p GPU --mem=12G -t 0 -w compute-0-7`
1. `ssh compute-0-7` Change the compute number as required
2. `cd /scratch/sashank/11785_project`
3. ` rsync -av sashank@128.2.176.255:/media/YertleDrive4/layer_grasp/dataset ./`
4. Now the datasets are synced up. We need to sync up the code `cd /home/stirumal/ws/cloth_segmentation_proj`, `git pull origin main`
5. Finally run the training code with `sbatch cloth_segmentation`.
