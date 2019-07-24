#!/bin/bash

#for SGE jobs
SCRIPTS='/home/kahwang/bkh/bin/Clock/'

mkdir $SCRIPTS/tmp

for ch in $(cat ${SCRIPTS}/channels); do

	echo "generaating script for channel $ch"

	echo "echo ${ch} | python $SCRIPTS/Clock.py" > $SCRIPTS/tmp/$ch.sh

	#INSERT YOUR SYSTEMS SUB, like
	#qsub -V $SCRIPTS/tmp/$id.sh

done

