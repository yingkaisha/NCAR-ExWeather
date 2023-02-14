
# general tools

import sys
from glob import glob

import numpy as np

sys.path.insert(0, '/glade/u/home/ksha/NCAR/')
sys.path.insert(0, '/glade/u/home/ksha/NCAR/libs/')

from namelist import *
import data_utils as du

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('lead', help='lead')
args = vars(parser.parse_args())

# =============== #
lead = int(args['lead'])

filenames = sorted(glob("/glade/campaign/cisl/aiml/ksha/NCAR_batch/TRAIN*neg_neg_neg*lead{}.npy".format(lead))) + \
            sorted(glob("/glade/campaign/cisl/aiml/ksha/NCAR_batch/TRAIN*pos*lead{}.npy".format(lead)))

#filenames = filenames[:200]

L = len(filenames)
out = np.empty((L, 15))

for i, name in enumerate(filenames):
    data = np.load(name)
    out[i, :] = data[0, 31, 31, :]

np.save("/glade/work/ksha/NCAR/TRAIN_pp15_single_lead{}.npy".format(lead), out)





