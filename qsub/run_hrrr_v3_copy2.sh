#!/bin/bash -l

#PBS -N HRRRv3
#PBS -A NAML0001
#PBS -l walltime=23:59:59
#PBS -l select=1:ncpus=4:ngpus=1:mem=12GB
#PBS -q casper
#PBS -o hrrr_v3_copy2.log
#PBS -e hrrr_v3_copy2.err

which python

cd /glade/u/home/ksha/NCAR/scripts/
#python HRRRv3_subset.py 06
#python HRRRv3_subset.py 09
#python HRRRv3_subset.py 12
#python HRRRv3_subset.py 15
#python HRRRv3_subset.py 18
#python HRRRv3_subset.py 01
#python HRRRv3_subset.py 02
python HRRRv3_subset.py 23
#python HRRRv3_subset.py 22
#python HRRRv3_subset.py 21
