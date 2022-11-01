#!/bin/bash -l

#PBS -N train0
#PBS -A NAML0001
#PBS -l walltime=23:59:59
#PBS -l select=1:ncpus=4:ngpus=1:mem=12GB
#PBS -l gpu_type=v100
#PBS -q casper
#PBS -o train0.out
#PBS -e train0.err

module load cuda/11.0.3
module load cudnn/8.0.4.30

#ncar_pylib ncar_20200417
#export PATH="/glade/work/${USER}/py_env_20200417/bin:$PATH"

cd /glade/u/home/ksha/NCAR/scripts/
python TRAIN_INF_vector.py 0 1
python TRAIN_INF_vector.py 0 3
python TRAIN_INF_vector.py 0 4

