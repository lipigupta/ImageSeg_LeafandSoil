#!/bin/bash
#SBATCH -q debug
#SBATCH -A nstaff_g
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH --gpus-per-node=4
#SBATCH --time=00:20:00
#SBATCH -J image_seg

module load cudatoolkit
source activate myPyTorch

python train.py
