#!/bin/bash

#for SGE jobs
SCRIPTS='/home/despoB/kaihwang/bin/Clock'


#run_mriqc.sh or run_fmriprep.sh

submit \
-s ${SCRIPTS}/run_indiv.sh \
-f ${SCRIPTS}/subjects \
-o ${SCRIPTS}/qsub.options




