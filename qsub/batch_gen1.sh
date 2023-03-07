#!/bin/bash -l

#PBS -N batch_g1
#PBS -A NAML0001
#PBS -l walltime=23:59:59
#PBS -l select=1:ncpus=4:mem=32GB
#PBS -q casper
#PBS -o batch_gen1.log
#PBS -e batch_gen1.err

which python

cd /glade/u/home/ksha/NCAR/scripts/

python DATA03_BATCH_gen_v3.py 2
# python DATA03_BATCH_gen_v4.py 2
# python DATA03_BATCH_gen_v4x.py 2

