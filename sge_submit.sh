#!/bin/bash

#for SGE jobs
SCRIPTS='/gpfs/group/mnh5174/default/Michael/Clock_MEG/Hwang_Clock'

mkdir $SCRIPTS/tmp

for ch in $(cat ${SCRIPTS}/channels); do
    
    echo "submitting script for channel $ch"
    qsub -v channel=$ch qsub_one_channel.bash

done

