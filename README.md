# TAC
Repository for TAC project

# Anaconda Environment Setup
```
# Create a new conda environment containing the GPU version of TensorFlow
conda create -n tf-gpu tensorflow-gpu

# As of April 2024, the GPU version of TensorFlow installed by the previous line is 2.4.1. Because we are using an older version of TensorFlow, we will need to downgrade numpy.
conda install numpy=1.19.5 --name tf-gpu

# The following lines will install other packages used by our scripts, specifically glob2 and keras-tuner. Because we are using an older version of TensorFlow, we select a compatible version of keras-tuner.
conda install conda-forge::glob2 --name tf-gpu
conda create --name kt --clone tf-gpu
conda install conda-forge::keras-tuner=1.1.0 --name kt
conda install keras-gpu --name kt
```

# Using Emory HyPER Community Cloud Cluster (C3)
## Migrating Data
To transfer data to and from the cluster, first connect to the Heart AI Lab's storage server (smb://nasn2acts.cc.emory.edu/heartailab-ts). Next, open a terminal window and initiate a Secure File Transfer Protocol (`sftp`) session.
```
sftp <Emory_NetID>@cirrostratus.it.emory.edu
```
It is highly recommended that users store their input and output data in the `/scratch` directory on the cluster, but note that the contents of this directory are purged after 60 days.
```
sftp> cd /scratch/<Emory_NetID>
```
To move files from the Heart AI Lab's storage server to your `/scratch` directory on the cluster...
```
sftp> put /path/to/file/on/heartailab-ts
```
To move files from your `/scratch` directory on the cluster to your local machine...
```
sftp> get /path/to/file/on/HyPER_C3
```
To exit the `sftp ` session...
```
sftp> quit
```
## Submitting scripts
On the HyPER C3, `sbatch` submits a batch script to SLURM for job management. The basic anatomy of a batch script you can submit to SLURM is below:
```
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --job-name=testGPU
#SBATCH --output=%x_%j.out
#SBATCH --account=general
#SBATCH --partition=a10g-1-gm24-c4-m16 
#SBATCH --mem=2G
#SBATCH --gpus=1

conda init bash>/dev/null 2>&1

source ~/.bashrc

conda activate kt

python example.py
```
Where `kt` is the conda environment we created above and `example.py` is a python script you would like to submit. To decide which `partition` to submit your script, use `squeue` to view which partitions are currently active on the HyPER C3. When you first login to the cluster using `ssh <Emory_NetID>@cirrostratus.it.emory.edu`, all partitions on the HyPER C3 will be listed along with the memory of their GPUs. Edit the batch script above to select your desired `partition` and your memory (`mem`) requirement to run your python script (`example.py`). When you are ready to submit your script, use...
```
sbatch batch_script.sh
```
