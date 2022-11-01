#!/bin/bash -l

#PBS -N stats
#PBS -A NAML0001
#PBS -l walltime=10:59:59
#PBS -l select=1:ncpus=4:ngpus=1:mem=12GB
#PBS -q casper
#PBS -o valid_gen.log
#PBS -e valid_gen.err

which python

cd /glade/u/home/ksha/NCAR/scripts/

python valid_gen.py
