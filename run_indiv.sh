#!/bin/bash
export DISPLAY=""

#Subject=${SGE_TASK}

#qsub -l mem_free=15G -V -M kaihwang -m e -e ~/tmp -o ~/tmp run_indiv.sh
python /home/despoB/kaihwang/bin/Clock/Clock.py

#echo "$Subject" | python /home/despoB/kaihwang/bin/Clock/Clock.py
#qsub -l mem_free=7G -binding linear:3 -pe threaded 3 -V -M kaihwang -m e -e ~/tmp -o ~/tmp ${SCRIPT}/python_qsub.sh


#cd /home/despoB/connectome-data
# for s in $(cat ~/bin/ThaGate/HCP_subjlist); do  #$(/bin/ls -d *)
# 	#if [ ! -e "/home/despoB/kaihwang/Rest/Graph/gsetCI_${Subject}.mat" ]; then
# 	sed "s/s in 100307/s in ${s}/g" < ${SCRIPT}/python_qsub.sh > ~/tmp/dFC_graph${s}.sh
# 	qsub -l mem_free=7G -V -M kaihwang -m e -e ~/tmp -o ~/tmp ~/tmp/dFC_graph${s}.sh
# 	#fi
# done

