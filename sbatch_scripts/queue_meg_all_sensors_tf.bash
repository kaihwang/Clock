#!/bin/bash

epochs="RT clock"
tmin=-2
tmax=2

for e in $epochs; do
    while read -r sensor; do	
	sbatch --export=epoch=${e},sensor=$sensor,tmin="$tmin",tmax="$tmax" sbatch_meg_one_sensor_tf.bash
    done < ../meg_sensors.txt
done

