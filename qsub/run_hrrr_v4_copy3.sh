#!/bin/bash -l

#PBS -N HRRRv4
#PBS -A NAML0001
#PBS -l walltime=23:59:59
#PBS -l select=1:ncpus=4:ngpus=1:mem=12GB
#PBS -q casper
#PBS -o hrrr_v4_04.log
#PBS -e hrrr_v4_04.err

which python

cd /glade/u/home/ksha/NCAR/scripts/

python HRRRv4_subset.py 04

