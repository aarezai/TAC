#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --job-name=descAo
#SBATCH --output=%x_%j.out
#SBATCH --partition=a10g-4-gm96-c48-m192
#SBATCH --gpus=1
#SBATCH --mem=50G

#SBATCH --mail-user mhalice@emory.edu
#SBATCH --mail-type BEGIN
#SBATCH --mail-type END
#SBATCH --mail-type FAIL


echo "Job started at "`date`

conda init bash>/dev/null 2>&1

source ~/.bashrc

conda activate tf_gpu

python DescAo_3DUnet_v1.py


echo "Job done at "`date`
