#!/bin/bash -l
#SBATCH --partition=a100
#SBATCH --job-name=MoE
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=1
#SBATCH -o /home/hpc/ptfs/ptfs325h/MoE-for-Denoising/output/slurm-%j.out
#SBATCH -e /home/hpc/ptfs/ptfs325h/MoE-for-Denoising/output/slurm-%j.err
#
# do no export environment variables
#SBATCH --export=NONE
#
# do not export environment variables
unset SLURM_EXPORT_ENV
#
# load required modules
module load python
#
# anaconda environment
source activate pyronn --num_epochs 10 --batch_size 32 --learning_rate 1e-3
#
# configure paths
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
#
# run
python train.py