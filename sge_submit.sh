#!/bin/bash

#for SGE jobs
SCRIPTS='/home/despoB/kaihwang/bin/Clock'

cat ${SCRIPTS}/channels

for ch in MEG2441; do

	id=$(echo ${ch} | grep -oE [0-9]{4})
	
	sed "s/MEG0713/${ch}/g" < ${SCRIPTS}/run_indiv.sh > ~/tmp/tfreg_${id}.sh

	submit \
	-s ~/tmp/tfreg_${id}.sh \
	-f ${SCRIPTS}/fullfreqs \
	-o ${SCRIPTS}/qsub.options


done

