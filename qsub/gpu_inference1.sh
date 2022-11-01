#!/bin/bash -l

#PBS -N cnn_pred69
#PBS -A NAML0001
#PBS -l walltime=23:59:59
#PBS -l select=1:ncpus=4:ngpus=1:mem=12GB
#PBS -l gpu_type=v100
#PBS -q casper
#PBS -o cnn_pred69.out
#PBS -e cnn_pred69.err

module load cuda/11.0.3
module load cudnn/8.0.4.30

cd /glade/u/home/ksha/NCAR/scripts/

python HRRR_inference.py 6
python HRRR_inference.py 9
