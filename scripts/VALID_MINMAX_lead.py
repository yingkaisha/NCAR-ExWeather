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

filename_valid_lead = sorted(glob("/glade/scratch/ksha/DATA/NCAR_batch_v3/VALID*lead{}.npy".format(lead)))

IND_lead = np.load('/glade/work/ksha/NCAR/IND_VALID_lead.npy', allow_pickle=True)[()]

L = len(IND_lead['lead{}'.format(lead)])

VALID_MAX = np.empty((L, 15))
VALID_MEAN = np.empty((L, 15))

start_time = time.time()

for i in range(L):
    ind_lead = int(IND_lead['lead{}'.format(lead)][i])
    filename_pick = filename_valid_lead[ind_lead]
    
    for c in range(15):
        temp_data = np.load(filename_pick)[..., c]
        
        VALID_MAX[i, c] = np.max(temp_data)
        VALID_MEAN[i, c] = np.mean(temp_data)
        
print("--- %s seconds ---" % (time.time() - start_time))

save_dict = {}
save_dict['MAX'] = VALID_MAX
save_dict['MEAN'] = VALID_MEAN

np.save('/glade/work/ksha/NCAR/VALID_MINMAX_lead{}.npy'.format(lead), save_dict)
print('/glade/work/ksha/NCAR/VALID_MINMAX_lead{}.npy'.format(lead))
