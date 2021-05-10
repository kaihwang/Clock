#!/bin/bash

epochs="RT clock"
basedir="/proj/mnhallqlab/projects/Clock_MEG/dan_source_rds"

for e in $epochs; do
    slist=$( find ${basedir}/${e}_time -type f -iname "*source.rds" )
    for ss in $slist; do
	sbatch --export=epoch=${e},sourcefile="$ss" sbatch_meg_one_source_tf.bash
    done
done

