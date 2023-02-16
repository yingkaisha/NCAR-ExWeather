
import numpy as np

prefix = '/glade/work/ksha/NCAR/TEST_lead_inds_v4_part{}.npy'

TRAIN_inds_p0  = np.load(prefix.format(0), allow_pickle=True)[()]
TRAIN_inds_p1  = np.load(prefix.format(1), allow_pickle=True)[()]
TRAIN_inds_p2  = np.load(prefix.format(2), allow_pickle=True)[()]
TRAIN_inds_p3  = np.load(prefix.format(3), allow_pickle=True)[()]
TRAIN_inds_p4  = np.load(prefix.format(4), allow_pickle=True)[()]
TRAIN_inds_p5  = np.load(prefix.format(5), allow_pickle=True)[()]
TRAIN_inds_p6  = np.load(prefix.format(6), allow_pickle=True)[()]
TRAIN_inds_p7  = np.load(prefix.format(7), allow_pickle=True)[()]
TRAIN_inds_p8  = np.load(prefix.format(8), allow_pickle=True)[()]
TRAIN_inds_p9  = np.load(prefix.format(9), allow_pickle=True)[()]
TRAIN_inds_p10  = np.load(prefix.format(10), allow_pickle=True)[()]
TRAIN_inds_p11  = np.load(prefix.format(11), allow_pickle=True)[()]

TRAIN_ind2 = np.concatenate((TRAIN_inds_p0['ind2'], TRAIN_inds_p1['ind2'], TRAIN_inds_p2['ind2'], TRAIN_inds_p3['ind2'],
                             TRAIN_inds_p4['ind2'], TRAIN_inds_p5['ind2'], TRAIN_inds_p6['ind2'], TRAIN_inds_p7['ind2'],
                             TRAIN_inds_p8['ind2'], TRAIN_inds_p9['ind2'], TRAIN_inds_p10['ind2'], TRAIN_inds_p11['ind2'],))

TRAIN_ind3 = np.concatenate((TRAIN_inds_p0['ind3'], TRAIN_inds_p1['ind3'], TRAIN_inds_p2['ind3'], TRAIN_inds_p3['ind3'],
                             TRAIN_inds_p4['ind3'], TRAIN_inds_p5['ind3'], TRAIN_inds_p6['ind3'], TRAIN_inds_p7['ind3'],
                             TRAIN_inds_p8['ind3'], TRAIN_inds_p9['ind3'], TRAIN_inds_p10['ind3'], TRAIN_inds_p11['ind3'],))

TRAIN_ind4 = np.concatenate((TRAIN_inds_p0['ind4'], TRAIN_inds_p1['ind4'], TRAIN_inds_p2['ind4'], TRAIN_inds_p3['ind4'],
                             TRAIN_inds_p4['ind4'], TRAIN_inds_p5['ind4'], TRAIN_inds_p6['ind4'], TRAIN_inds_p7['ind4'],
                             TRAIN_inds_p8['ind4'], TRAIN_inds_p9['ind4'], TRAIN_inds_p10['ind4'], TRAIN_inds_p11['ind4'],))

TRAIN_ind5 = np.concatenate((TRAIN_inds_p0['ind5'], TRAIN_inds_p1['ind5'], TRAIN_inds_p2['ind5'], TRAIN_inds_p3['ind5'],
                             TRAIN_inds_p4['ind5'], TRAIN_inds_p5['ind5'], TRAIN_inds_p6['ind5'], TRAIN_inds_p7['ind5'],
                             TRAIN_inds_p8['ind5'], TRAIN_inds_p9['ind5'], TRAIN_inds_p10['ind5'], TRAIN_inds_p11['ind5'],))

TRAIN_ind6 = np.concatenate((TRAIN_inds_p0['ind6'], TRAIN_inds_p1['ind6'], TRAIN_inds_p2['ind6'], TRAIN_inds_p3['ind6'],
                             TRAIN_inds_p4['ind6'], TRAIN_inds_p5['ind6'], TRAIN_inds_p6['ind6'], TRAIN_inds_p7['ind6'],
                             TRAIN_inds_p8['ind6'], TRAIN_inds_p9['ind6'], TRAIN_inds_p10['ind6'], TRAIN_inds_p11['ind6'],))



TRAIN_ind20 = np.concatenate((TRAIN_inds_p0['ind20'], TRAIN_inds_p1['ind20'], TRAIN_inds_p2['ind20'], TRAIN_inds_p3['ind20'],
                              TRAIN_inds_p4['ind20'], TRAIN_inds_p5['ind20'], TRAIN_inds_p6['ind20'], TRAIN_inds_p7['ind20'],
                              TRAIN_inds_p8['ind20'], TRAIN_inds_p9['ind20'], TRAIN_inds_p10['ind20'], TRAIN_inds_p11['ind20'],))

TRAIN_ind21 = np.concatenate((TRAIN_inds_p0['ind21'], TRAIN_inds_p1['ind21'], TRAIN_inds_p2['ind21'], TRAIN_inds_p3['ind21'],
                              TRAIN_inds_p4['ind21'], TRAIN_inds_p5['ind21'], TRAIN_inds_p6['ind21'], TRAIN_inds_p7['ind21'],
                              TRAIN_inds_p8['ind21'], TRAIN_inds_p9['ind21'], TRAIN_inds_p10['ind21'], TRAIN_inds_p11['ind21'],))

TRAIN_ind22 = np.concatenate((TRAIN_inds_p0['ind22'], TRAIN_inds_p1['ind22'], TRAIN_inds_p2['ind22'], TRAIN_inds_p3['ind22'],
                              TRAIN_inds_p4['ind22'], TRAIN_inds_p5['ind22'], TRAIN_inds_p6['ind22'], TRAIN_inds_p7['ind22'],
                              TRAIN_inds_p8['ind22'], TRAIN_inds_p9['ind22'], TRAIN_inds_p10['ind22'], TRAIN_inds_p11['ind22'],))

TRAIN_ind23 = np.concatenate((TRAIN_inds_p0['ind23'], TRAIN_inds_p1['ind23'], TRAIN_inds_p2['ind23'], TRAIN_inds_p3['ind23'],
                              TRAIN_inds_p4['ind23'], TRAIN_inds_p5['ind23'], TRAIN_inds_p6['ind23'], TRAIN_inds_p7['ind23'],
                              TRAIN_inds_p8['ind23'], TRAIN_inds_p9['ind23'], TRAIN_inds_p10['ind23'], TRAIN_inds_p11['ind23'],))

IND_TRAIN_lead = {}
IND_TRAIN_lead['lead2'] = TRAIN_ind2
IND_TRAIN_lead['lead3'] = TRAIN_ind3
IND_TRAIN_lead['lead4'] = TRAIN_ind4
IND_TRAIN_lead['lead5'] = TRAIN_ind5
IND_TRAIN_lead['lead6'] = TRAIN_ind6
IND_TRAIN_lead['lead20'] = TRAIN_ind20
IND_TRAIN_lead['lead21'] = TRAIN_ind21
IND_TRAIN_lead['lead22'] = TRAIN_ind22
IND_TRAIN_lead['lead23'] = TRAIN_ind23


np.save('/glade/work/ksha/NCAR/IND_TEST_lead_v4.npy', IND_TRAIN_lead)
print('/glade/work/ksha/NCAR/IND_TEST_lead_v4.npy')

# np.save('/glade/work/ksha/NCAR/IND_VALID_lead_v4x.npy', IND_VALID_lead)
# print('/glade/work/ksha/NCAR/IND_VALID_lead_full.npy')

