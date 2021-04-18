#!/bin/bash

#SBATCH -p general
#SBATCH -N 1
#SBATCH --mem 192g
#SBATCH -n 8
#SBATCH -t 12:00:00
#SBATCH --mail-type=all
#SBATCH --mail-user=mnhallq@email.unc.edu

module use /proj/mnhallqlab/sw/modules
module load r/4.0.3_depend

export epoch=RT

R CMD BATCH --no-save --no-restore combine_meg_evoked.R
R CMD BATCH --no-save --no-restore subsample_meg.R
R CMD BATCH --no-save --no-restore time_freq.R
