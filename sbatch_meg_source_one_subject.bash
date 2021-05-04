#!/bin/bash

#SBATCH -p general
#SBATCH -N 1
#SBATCH --mem 92g
#SBATCH -n 1
#SBATCH -t 2:00:00

#module load anaconda/2020.07
. "/nas/longleaf/apps/anaconda/2020.07/etc/profile.d/conda.sh"
conda activate mne
export subject epoch
python3 apply_inverse.py

