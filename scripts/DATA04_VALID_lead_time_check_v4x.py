# general tools
import os
import re
import sys
import time
import random
from glob import glob

import numpy as np

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('part', help='part')
args = vars(parser.parse_args())

# =============== #
part = int(args['part'])

def id_extract(filenames):
    
    indx_out = []
    indy_out = []
    lead_out = []
    day_out = []
    
    for i, name in enumerate(filenames):        
        nums = re.findall(r'\d+', name)
        lead = int(nums[-1])
        indy = int(nums[-2])
        indx = int(nums[-3])
        day = int(nums[-4])
                
        indx_out.append(indx)
        indy_out.append(indy)
        lead_out.append(lead)
        day_out.append(day)
        
    return np.array(indx_out), np.array(indy_out), np.array(lead_out), np.array(day_out)

# import filenames

filename_valid_lead2 = sorted(glob("/glade/campaign/cisl/aiml/ksha/NCAR_batch_v4x/VALID*lead2.npy"))
filename_valid_lead3 = sorted(glob("/glade/campaign/cisl/aiml/ksha/NCAR_batch_v4x/VALID*lead3.npy"))
filename_valid_lead4 = sorted(glob("/glade/campaign/cisl/aiml/ksha/NCAR_batch_v4x/VALID*lead4.npy"))
filename_valid_lead5 = sorted(glob("/glade/campaign/cisl/aiml/ksha/NCAR_batch_v4x/VALID*lead5.npy"))
filename_valid_lead6 = sorted(glob("/glade/campaign/cisl/aiml/ksha/NCAR_batch_v4x/VALID*lead6.npy"))

indx2, indy2, lead2, day2 = id_extract(filename_valid_lead2)
indx3, indy3, lead3, day3 = id_extract(filename_valid_lead3)
indx4, indy4, lead4, day4 = id_extract(filename_valid_lead4)
indx5, indy5, lead5, day5 = id_extract(filename_valid_lead5)
indx6, indy6, lead6, day6 = id_extract(filename_valid_lead6)

L = len(filename_valid_lead2)

ind2 = np.empty(L); ind3 = np.empty(L); ind4 = np.empty(L); ind5 = np.empty(L)
ind6 = np.empty(L)

count = 0

start_time = time.time()

gap = 20000
ind_end = (part+1)*gap
ind_end_min = np.min([L, ind_end])

picks = range(part*gap, ind_end_min, 1)

rad = 241800

for i in picks:

    i_start = np.max([i-rad, 0])
    i_end = np.min([i+rad, L])
    
    ind_temp2 = np.nan; ind_temp3 = np.nan; ind_temp4 = np.nan
    ind_temp5 = np.nan; ind_temp6 = np.nan

    pattern_day = 'VALID_day{:03d}'.format(day2[i])

    patten_lead3 = 'indx{}_indy{}_lead3'.format(indx2[i], indy2[i])
    patten_lead4 = 'indx{}_indy{}_lead4'.format(indx2[i], indy2[i])
    patten_lead5 = 'indx{}_indy{}_lead5'.format(indx2[i], indy2[i])
    patten_lead6 = 'indx{}_indy{}_lead6'.format(indx2[i], indy2[i])

    for i3, name3 in enumerate(filename_valid_lead3[i_start:i_end]):
        if (pattern_day in name3) and (patten_lead3 in name3):
            ind_temp3 = i_start+i3
            break;

    for i4, name4 in enumerate(filename_valid_lead4[i_start:i_end]):
        if (pattern_day in name4) and (patten_lead4 in name4):
            ind_temp4 = i_start+i4
            break;

    for i5, name5 in enumerate(filename_valid_lead5[i_start:i_end]):
        if (pattern_day in name5) and (patten_lead5 in name5):
            ind_temp5 = i_start+i5
            break;

    for i6, name6 in enumerate(filename_valid_lead6[i_start:i_end]):
        if (pattern_day in name6) and (patten_lead6 in name6):
            ind_temp6 = i_start+i6
            break;

    flag = ind_temp3
    flag = flag + ind_temp4 + ind_temp5 + ind_temp6

    if np.logical_not(np.isnan(flag)): 
        ind2[count] = i
        ind3[count] = ind_temp3; ind4[count] = ind_temp4; ind5[count] = ind_temp5
        ind6[count] = ind_temp6
        count += 1

print("--- %s seconds ---" % (time.time() - start_time))

print(count)
save_dict = {}
save_dict['ind2'] = ind2[:count]
save_dict['ind3'] = ind3[:count]
save_dict['ind4'] = ind4[:count]
save_dict['ind5'] = ind5[:count]
save_dict['ind6'] = ind6[:count]

np.save('/glade/work/ksha/NCAR/VALID_lead_inds_v4x_part{}.npy'.format(part), save_dict)
print('/glade/work/ksha/NCAR/VALID_lead_inds_v4x_part{}.npy'.format(part))
    
