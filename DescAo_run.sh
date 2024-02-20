#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --job-name=descAo
#SBATCH --output=%x_%j.out
#SBATCH --partition=a100-8-gm320-c96-m1152
#SBATCH --gpus=1
#SBATCH --mem=160G

#SBATCH --mail-type BEGIN
#SBATCH --mail-type END
#SBATCH --mail-type FAIL


echo "Job started at "`date`

conda init bash>/dev/null 2>&1

source ~/.bashrc

conda activate tf_gpu

python DescAo_3DUnet_v1.py


echo "Job done at "`date`
