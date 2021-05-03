#!/bin/bash
set -ex

while read this_subject
do
    sbatch --export=subject=${this_subject} sbatch_meg_source_one_subject.bash
done < subjects
