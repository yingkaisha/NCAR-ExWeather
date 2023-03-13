
import numpy as np

data_dir = '/glade/work/ksha/NCAR/'
report_dir = data_dir+'storm_report/'
NCAR500_dir = '/glade/p/mmm/parc/sobash/NSC/3KM_WRF_POST_12sec_ts/'
#'/glade/work/sobash/NSC_objects/HRRR_new' 
#'/glade/u/home/sobash/2013RT/REPORTS/reports_v20221101.db'

# File path
filepath_vec = "/glade/work/ksha/NCAR/"
# HRRR v3
path_batch_v3 = '/glade/campaign/cisl/aiml/ksha/NCAR_batch_v3/'
# HRRR v4x
path_batch_v4x = '/glade/campaign/cisl/aiml/ksha/NCAR_batch_v4x/'
# HRRR v4
path_batch_v4 = '/glade/campaign/cisl/aiml/ksha/NCAR_batch_v4_temp/'

HRRR_dir = '/glade/scratch/ksha/DRIVE/HRRR/'
save_dir = '/glade/work/ksha/NCAR/'
save_dir_scratch = '/glade/scratch/ksha/DRIVE/'
save_dir_campaign = '/glade/campaign/cisl/aiml/ksha/'

data_pd500_dir = '/glade/scratch/ksha/DRIVE/clean_NCAR/'
data_pdhrrr_dir = '/glade/scratch/ksha/DRIVE/clean_HRRR/'

# figure storage
fig_dir = '/glade/u/home/ksha/figures/'

# Matplotlib figure export settings
fig_keys = {'dpi':250, 
            'orientation':'portrait', 
            'bbox_inches':'tight', 
            'pad_inches':0.1, 
            'transparent':False}

# colors
#
rgb_array = np.array([[0.85      , 0.85      , 0.85      , 1.        ],
                      [0.66666667, 1.        , 1.        , 1.        ],
                      [0.33333333, 0.62745098, 1.        , 1.        ],
                      [0.11372549, 0.        , 1.        , 1.        ],
                      [0.37647059, 0.81176471, 0.56862745, 1.        ],
                      [0.10196078, 0.59607843, 0.31372549, 1.        ],
                      [0.56862745, 0.81176471, 0.37647059, 1.        ],
                      [0.85098039, 0.9372549 , 0.54509804, 1.        ],
                      [1.        , 1.        , 0.4       , 1.        ],
                      [1.        , 0.8       , 0.4       , 1.        ],
                      [1.        , 0.53333333, 0.29803922, 1.        ],
                      [1.        , 0.09803922, 0.09803922, 1.        ],
                      [0.8       , 0.23921569, 0.23921569, 1.        ],
                      [0.64705882, 0.19215686, 0.19215686, 1.        ],
                      [0.55      , 0.        , 0.        , 1.        ]])
    
blue   = rgb_array[3, :]  # blue
cyan   = rgb_array[2, :]  # cyan
lgreen = rgb_array[4, :]  # light green
green  = rgb_array[5, :]  # dark green
yellow = rgb_array[8, :]  # yellow
orange = rgb_array[-6, :] # orange
red    = rgb_array[-3, :] # red