#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --job-name=hyperParamTuning
#SBATCH --output=%x_%j.out
#SBATCH --partition=a10g-8-gm192-c192-m768
#SBATCH --gpus=1
#SBATCH --mem=175G

echo "Job started at "`date`

conda init bash>/dev/null 2>&1

source ~/.bashrc

conda activate kt

python DescAo_3DUnet_v1_Tuned.py 


echo "Job done at "`date`
