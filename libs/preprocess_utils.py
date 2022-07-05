
import numpy as np

def frame_selector(data, xi, yi, gap_target, N_range_encode):
    '''
    Selecting [gap_target*N_range_encode] sized patches from 3-km model run
    '''
    assert N_range_encode % 2 == 1
    N_range_half = (N_range_encode - 1) // 2 # half of the encoded target grids
    gap_encode_left = gap_target*N_range_half # left side of the encoded target grids converted to model grids
    gap_encode_right = gap_target*N_range_half+gap_target # right side of the encoded target grids converted to model grids
    return data[xi*gap_target-gap_encode_left:xi*gap_target+gap_encode_right, yi*gap_target-gap_encode_left:yi*gap_target+gap_encode_right]
