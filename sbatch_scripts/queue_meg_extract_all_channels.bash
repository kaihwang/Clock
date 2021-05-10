#!/bin/bash
set -ex

while read this_channel
do
    sbatch --export=channel=${this_channel} sbatch_meg_extract_one_channel.bash
done < channels
