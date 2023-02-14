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

filename_train_lead2 = sorted(glob("/glade/scratch/ksha/DATA/NCAR_batch_v3/TRAIN*neg_neg_neg*lead2.npy")) + \
                       sorted(glob("/glade/scratch/ksha/DATA/NCAR_batch_v3/TRAIN*pos*lead2.npy"))

filename_train_lead3 = sorted(glob("/glade/scratch/ksha/DATA/NCAR_batch_v3/TRAIN*neg_neg_neg*lead3.npy")) + \
                       sorted(glob("/glade/scratch/ksha/DATA/NCAR_batch_v3/TRAIN*pos*lead3.npy"))

filename_train_lead4 = sorted(glob("/glade/scratch/ksha/DATA/NCAR_batch_v3/TRAIN*neg_neg_neg*lead4.npy")) + \
                       sorted(glob("/glade/scratch/ksha/DATA/NCAR_batch_v3/TRAIN*pos*lead4.npy"))

filename_train_lead5 = sorted(glob("/glade/scratch/ksha/DATA/NCAR_batch_v3/TRAIN*neg_neg_neg*lead5.npy")) + \
                       sorted(glob("/glade/scratch/ksha/DATA/NCAR_batch_v3/TRAIN*pos*lead5.npy"))

filename_train_lead6 = sorted(glob("/glade/scratch/ksha/DATA/NCAR_batch_v3/TRAIN*neg_neg_neg*lead6.npy")) + \
                       sorted(glob("/glade/scratch/ksha/DATA/NCAR_batch_v3/TRAIN*pos*lead6.npy"))

filename_train_lead7 = sorted(glob("/glade/scratch/ksha/DATA/NCAR_batch_v3/TRAIN*neg_neg_neg*lead7.npy")) + \
                       sorted(glob("/glade/scratch/ksha/DATA/NCAR_batch_v3/TRAIN*pos*lead7.npy"))

filename_train_lead8 = sorted(glob("/glade/scratch/ksha/DATA/NCAR_batch_v3/TRAIN*neg_neg_neg*lead8.npy")) + \
                       sorted(glob("/glade/scratch/ksha/DATA/NCAR_batch_v3/TRAIN*pos*lead8.npy"))

filename_train_lead19 = sorted(glob("/glade/scratch/ksha/DATA/NCAR_batch_v3/TRAIN*neg_neg_neg*lead19.npy")) + \
                        sorted(glob("/glade/scratch/ksha/DATA/NCAR_batch_v3/TRAIN*pos*lead19.npy"))

filename_train_lead20 = sorted(glob("/glade/scratch/ksha/DATA/NCAR_batch_v3/TRAIN*neg_neg_neg*lead20.npy")) + \
                        sorted(glob("/glade/scratch/ksha/DATA/NCAR_batch_v3/TRAIN*pos*lead20.npy"))

filename_train_lead21 = sorted(glob("/glade/scratch/ksha/DATA/NCAR_batch_v3/TRAIN*neg_neg_neg*lead21.npy")) + \
                        sorted(glob("/glade/scratch/ksha/DATA/NCAR_batch_v3/TRAIN*pos*lead21.npy"))

filename_train_lead22 = sorted(glob("/glade/scratch/ksha/DATA/NCAR_batch_v3/TRAIN*neg_neg_neg*lead22.npy")) + \
                        sorted(glob("/glade/scratch/ksha/DATA/NCAR_batch_v3/TRAIN*pos*lead22.npy"))

indx2, indy2, lead2, day2 = id_extract(filename_train_lead2)
indx3, indy3, lead3, day3 = id_extract(filename_train_lead3)
indx4, indy4, lead4, day4 = id_extract(filename_train_lead4)
indx5, indy5, lead5, day5 = id_extract(filename_train_lead5)
indx6, indy6, lead6, day6 = id_extract(filename_train_lead6)
indx7, indy7, lead7, day7 = id_extract(filename_train_lead7)
indx8, indy8, lead8, day8 = id_extract(filename_train_lead8)

indx19, indy19, lead19, day19 = id_extract(filename_train_lead19)
indx20, indy20, lead20, day20 = id_extract(filename_train_lead20)
indx21, indy21, lead21, day21 = id_extract(filename_train_lead21)
indx22, indy22, lead22, day22 = id_extract(filename_train_lead22)


L = len(filename_train_lead2)

ind2 = np.empty(L); ind3 = np.empty(L); ind4 = np.empty(L); ind5 = np.empty(L)
ind6 = np.empty(L); ind7 = np.empty(L); ind8 = np.empty(L); ind19 = np.empty(L)
ind20 = np.empty(L); ind21 = np.empty(L); ind22 = np.empty(L)

count = 0

start_time = time.time()

gap = 60000
ind_end = (part+1)*gap
ind_end_min = np.min([L, ind_end])

picks = range(part*gap, ind_end_min, 1)

for i in picks:

    ind_temp2 = np.nan; ind_temp3 = np.nan; ind_temp4 = np.nan
    ind_temp5 = np.nan; ind_temp6 = np.nan; ind_temp7 = np.nan
    ind_temp8 = np.nan; ind_temp19 = np.nan; ind_temp20 = np.nan
    ind_temp21 = np.nan; ind_temp22 = np.nan

    pattern_day = 'TRAIN_day{:03d}'.format(day2[i])

    patten_lead3 = 'indx{}_indy{}_lead3'.format(indx2[i], indy2[i])
    patten_lead4 = 'indx{}_indy{}_lead4'.format(indx2[i], indy2[i])
    patten_lead5 = 'indx{}_indy{}_lead5'.format(indx2[i], indy2[i])
    patten_lead6 = 'indx{}_indy{}_lead6'.format(indx2[i], indy2[i])
    patten_lead7 = 'indx{}_indy{}_lead7'.format(indx2[i], indy2[i])
    patten_lead8 = 'indx{}_indy{}_lead8'.format(indx2[i], indy2[i])
    patten_lead19 = 'indx{}_indy{}_lead19'.format(indx2[i], indy2[i])
    patten_lead20 = 'indx{}_indy{}_lead20'.format(indx2[i], indy2[i])
    patten_lead21 = 'indx{}_indy{}_lead21'.format(indx2[i], indy2[i])
    patten_lead22 = 'indx{}_indy{}_lead22'.format(indx2[i], indy2[i])

    for i3, name3 in enumerate(filename_train_lead3):
        if (pattern_day in name3) and (patten_lead3 in name3):
            ind_temp3 = i3
            break;

    for i4, name4 in enumerate(filename_train_lead4):
        if (pattern_day in name4) and (patten_lead4 in name4):
            ind_temp4 = i4
            break;

    for i5, name5 in enumerate(filename_train_lead5):
        if (pattern_day in name5) and (patten_lead5 in name5):
            ind_temp5 = i5
            break;

    for i6, name6 in enumerate(filename_train_lead6):
        if (pattern_day in name6) and (patten_lead6 in name6):
            ind_temp6 = i6
            break;

    for i7, name7 in enumerate(filename_train_lead7):
        if (pattern_day in name7) and (patten_lead7 in name7):
            ind_temp7 = i7
            break;

    for i8, name8 in enumerate(filename_train_lead8):
        if (pattern_day in name8) and (patten_lead8 in name8):
            ind_temp8 = i8
            break;

    for i19, name19 in enumerate(filename_train_lead19):
        if (pattern_day in name19) and (patten_lead19 in name19):
            ind_temp19 = i19
            break;

    for i20, name20 in enumerate(filename_train_lead20):
        if (pattern_day in name20) and (patten_lead20 in name20):
            ind_temp20 = i20
            break;

    for i21, name21 in enumerate(filename_train_lead21):
        if (pattern_day in name21) and (patten_lead21 in name21):
            ind_temp21 = i21
            break;

    for i22, name22 in enumerate(filename_train_lead22):
        if (pattern_day in name22) and (patten_lead22 in name22):
            ind_temp22 = i22
            break;

    flag = ind_temp3 + ind_temp4 + ind_temp5 + ind_temp6 + ind_temp7 + ind_temp8 + ind_temp19 + ind_temp20 + ind_temp21 + ind_temp22

    if np.logical_not(np.isnan(flag)): 
        ind2[count] = i
        ind3[count] = ind_temp3; ind4[count] = ind_temp4; ind5[count] = ind_temp5
        ind6[count] = ind_temp6; ind7[count] = ind_temp7; ind8[count] = ind_temp8
        ind19[count] = ind_temp19; ind20[count] = ind_temp20 
        ind21[count] = ind_temp21; ind22[count] = ind_temp22
        count += 1

print("--- %s seconds ---" % (time.time() - start_time))

print(count)
save_dict = {}
save_dict['ind2'] = ind2[:count]
save_dict['ind3'] = ind3[:count]
save_dict['ind4'] = ind4[:count]
save_dict['ind5'] = ind5[:count]
save_dict['ind6'] = ind6[:count]
save_dict['ind7'] = ind7[:count]
save_dict['ind8'] = ind8[:count]
save_dict['ind19'] = ind19[:count]
save_dict['ind20'] = ind20[:count]
save_dict['ind21'] = ind21[:count]
save_dict['ind22'] = ind22[:count]

np.save('/glade/work/ksha/NCAR/TRAIN_lead_inds_part{}.npy'.format(part), save_dict)
print('/glade/work/ksha/NCAR/TRAIN_lead_inds_part{}.npy'.format(part))
    
