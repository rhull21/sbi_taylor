import numpy
from parflowio.pyParflowio import PFData


def PFread(pfb_data):
    '''
    read in pfb 'data' and get numpy array
    '''
    pfb_data.loadHeader()
    pfb_data.loadData()
    data_arr = pfb_data.moveDataArray()
    pfb_data.close()
    return data_arr