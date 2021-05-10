#!/bin/bash
#set -ex

#this script queues the source localization for each subject using the apply_inverse.py script

epochs="clock RT"
outdir="/proj/mnhallqlab/projects/Clock_MEG/fif_data/csv_data"
for ee in $epochs; do
while read this_subject
do
    ofile="${outdir}/${ee}/${this_subject}_${ee}_source_ts.csv"
    if [ ! -f "$ofile" ]; then
        sbatch --export=subject=${this_subject},epoch=${ee} sbatch_meg_extract_one_subjectsource.bash
    else
        echo "file exists: $ofile"
    fi
done < ../subjects
done
