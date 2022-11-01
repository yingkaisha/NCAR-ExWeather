#!/bin/bash -l

#PBS -N stats
#PBS -A NAML0001
#PBS -l walltime=23:59:59
#PBS -l select=1:ncpus=4:ngpus=1:mem=12GB
#PBS -q casper
#PBS -o batch_full.log
#PBS -e batch_full.err

which python

cd /glade/u/home/ksha/NCAR/scripts/

python batch_gen_ncar.py 2

