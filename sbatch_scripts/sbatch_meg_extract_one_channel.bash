#!/bin/bash

#SBATCH -p general
#SBATCH -N 1
#SBATCH --mem 32g
#SBATCH -n 1
#SBATCH -t 3:00:00

#module load anaconda/2020.07
. "/nas/longleaf/apps/anaconda/2020.07/etc/profile.d/conda.sh"
conda activate mne
export channel
python3 save_evoke_to_df.py
