# general tools
import sys
from glob import glob
from datetime import datetime, timedelta

# data tools
import h5py
import numpy as np
import pandas as pd

# stats tools
from scipy.spatial import cKDTree
from scipy.interpolate import interp2d
from scipy.interpolate import NearestNDInterpolator
# from sklearn.metrics import mean_squared_error
# from sklearn.metrics import mean_absolute_error

def is_leap_year(year):
    '''
    Determine whether a year is a leap year.
    '''
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def save_hdf5(p_group, labels, out_dir, filename='example.hdf'):
    '''
    Save data into a signle hdf5
        - p_group: datasets combined in one tuple;
        - labels: list of strings;
        - out_dir: output path;
        - filename: example.hdf;
    **label has initial 'x' means ENCODED strings
    '''    
    name = out_dir+filename
    hdf = h5py.File(name, 'w')
    for i, label in enumerate(labels):
        if label[0] != 'x':
            hdf.create_dataset(label, data=p_group[i])
        else:
            string = p_group[i]
            hdf.create_dataset(label, (len(string), 1), 'S10', string)
    hdf.close()
    print('Save to {}'.format(name))


def grid_search(xgrid, ygrid, stn_lon, stn_lat):
    '''
    kdtree-based nearest gridpoint search
    output: indices_lon, indices_lat
    '''
    gridTree = cKDTree(list(zip(xgrid.ravel(), ygrid.ravel()))) #KDTree_wraper(xgrid, ygrid)
    grid_shape = xgrid.shape
    dist, indexes = gridTree.query(list(zip(stn_lon, stn_lat)))
    return np.unravel_index(indexes, grid_shape)

def interp2d_wraper(nav_lon, nav_lat, grid_z, out_lon, out_lat, method='linear'):
    '''
    wrapper of interp2d, works for 2-d grid to grid interp.
    method = 'linear' or 'cubic'
    output: grid
    '''
    if np.sum(np.isnan(grid_z)) > 0:
        grid_z = fill_coast_interp(grid_z, np.zeros(grid_z.shape).astype(bool))
        
    interp_obj = interp2d(nav_lon[0, :], nav_lat[:, 0], grid_z, kind=method)
    return interp_obj(out_lon[0, :], out_lat[:, 0])

def nearest_wraper(nav_lon, nav_lat, grid_z, out_lon, out_lat):
    '''
    wrapper of nearest neighbour
    '''
    f = NearestNDInterpolator((nav_lon.ravel(), nav_lat.ravel()), grid_z.ravel())
    out = f((out_lon.ravel(), out_lat.ravel()))
    return out.reshape(out_lon.shape)

