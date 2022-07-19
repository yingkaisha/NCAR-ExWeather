#!/bin/bash -l

#PBS -N HRRRv3
#PBS -A NAML0001
#PBS -l walltime=17:59:59
#PBS -l select=1:ncpus=4:ngpus=1:mem=12GB
#PBS -q casper
#PBS -o hrrr_v3.out
#PBS -e hrrr_v3.err

which python

cd /glade/u/home/ksha/NCAR/scripts/
python HRRR_test_collection_v3.py