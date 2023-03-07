#!/bin/bash -l

#PBS -N vec2
#PBS -A NAML0001
#PBS -l walltime=23:59:59
#PBS -l select=1:ncpus=4:ngpus=1:mem=12GB
#PBS -l gpu_type=v100
#PBS -q casper
#PBS -o vec2.out
#PBS -e vec2.err

module load cuda/11.0.3
module load cudnn/8.0.4.30

#ncar_pylib ncar_20200417
#export PATH="/glade/work/${USER}/py_env_20200417/bin:$PATH"

cd /glade/u/home/ksha/NCAR/scripts/

# python CNN00_feature_vector_training_set.py 0 4 RE2_peak_base5 peak v3
# python CNN00_feature_vector_training_set.py 1 4 RE2_peak_base5 peak v3
# python CNN00_feature_vector_training_set.py 2 4 RE2_peak_base5 peak v3
# python CNN00_feature_vector_validation_set.py 4 RE2_peak_base5 peak v3

python CNN00_feature_vector_training_32.py 0 4 RE2_vgg_base2 vgg v3
python CNN00_feature_vector_training_32.py 1 4 RE2_vgg_base2 vgg v3
python CNN00_feature_vector_training_32.py 2 4 RE2_vgg_base2 vgg v3
python CNN00_feature_vector_validation_32.py 4 RE2_vgg_base2 vgg v3

# python CNN00_feature_vector_training_set.py 0 4 RE2_peak_base5 peak v4x
# python CNN00_feature_vector_validation_set.py 4 RE2_peak_base5 peak v4x
