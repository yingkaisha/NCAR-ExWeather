#!/bin/bash -l

#PBS -N HRRRv3
#PBS -A NAML0001
#PBS -l walltime=23:59:59
#PBS -l select=1:ncpus=4:ngpus=1:mem=12GB
#PBS -q casper
#PBS -o hrrr_v32.log
#PBS -e hrrr_v32.err

which python

cd /glade/u/home/ksha/NCAR/scripts/

python HRRRv3_subset.py 05


