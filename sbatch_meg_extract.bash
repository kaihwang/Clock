#!/bin/bash

#SBATCH -p general
#SBATCH -N 1
#SBATCH --mem 32g
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH --mail-type=all
#SBATCH --mail-user=mnhallq@email.unc.edu

#module load anaconda/2020.07
. "/nas/longleaf/apps/anaconda/2020.07/etc/profile.d/conda.sh"
conda activate mne
python3 save_evoke_to_df.py
