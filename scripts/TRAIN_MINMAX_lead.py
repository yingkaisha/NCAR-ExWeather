# general tools
import time
from glob import glob
import numpy as np

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('lead', help='lead')
args = vars(parser.parse_args())
# =============== #
lead = int(args['lead'])

filename_train_lead = sorted(glob("/glade/scratch/ksha/DATA/NCAR_batch_v3/TRAIN*lead{}.npy".format(lead)))

IND_lead = np.load('/glade/work/ksha/NCAR/IND_TRAIN_lead.npy', allow_pickle=True)[()]

L = len(IND_lead['lead{}'.format(lead)])

TRAIN_MAX = np.empty((L, 15))
TRAIN_MEAN = np.empty((L, 15))

start_time = time.time()

for i in range(L):
    ind_lead = int(IND_lead['lead{}'.format(lead)][i])
    filename_pick = filename_train_lead[ind_lead]
    
    for c in range(15):
        temp_data = np.load(filename_pick)[..., c]
        
        TRAIN_MAX[i, c] = np.max(temp_data)
        TRAIN_MEAN[i, c] = np.mean(temp_data)
        
print("--- %s seconds ---" % (time.time() - start_time))

save_dict = {}
save_dict['MAX'] = TRAIN_MAX
save_dict['MEAN'] = TRAIN_MEAN

np.save('/glade/work/ksha/NCAR/TRAIN_MINMAX_lead{}.npy'.format(lead), save_dict)
print('/glade/work/ksha/NCAR/TRAIN_MINMAX_lead{}.npy'.format(lead))
