#!/bin/bash -l

#PBS -N HRRRv3
#PBS -A NAML0001
#PBS -l walltime=17:59:59
#PBS -l select=1:ncpus=4:ngpus=1:mem=12GB
#PBS -q casper
#PBS -o clean_hrrr.out
#PBS -e clean_hrrr.err

which python

cd /glade/u/home/ksha/NCAR/scripts/
python clean_HRRR_preprocess.py